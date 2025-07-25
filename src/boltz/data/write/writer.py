import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, Optional, List

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from boltz.data.types import Coords, Interface, Record, Structure, StructureV2
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
from boltz.data.write.trajectory import TrajectoryWriter
from boltz.data.write.unrotated_trajectory import create_unrotated_trajectory


class BoltzWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
        boltz2: bool = False,
        write_embeddings: bool = False,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.boltz2 = boltz2
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.write_embeddings = write_embeddings

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]

        # Get ranking
        if "confidence_score" in prediction:
            argsort = torch.argsort(prediction["confidence_score"], descending=True)
            idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}
        # Handles cases where confidence summary is False
        else:
            idx_to_rank = {i: i for i in range(len(records))}

        # Process and save structures first, then handle trajectory saving with final coordinates
        final_coords_for_trajectory = {}  # Store final coordinates for trajectory

        # Iterate over the records
        for record, coord, pad_mask in zip(records, coords, pad_masks):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            if self.boltz2:
                structure: StructureV2 = StructureV2.load(path)
            else:
                structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()
                
                # Store final coordinates for trajectory (using first model only)
                if model_idx == 0:
                    final_coords_for_trajectory[record.id] = {
                        'coords': coord_unpad.copy(),
                        'structure': structure,
                        'pad_mask': pad_mask
                    }

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True
                if self.boltz2:
                    structure: StructureV2
                    coord_unpad = [(x,) for x in coord_unpad]
                    coord_unpad = np.array(coord_unpad, dtype=Coords)

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                if self.boltz2:
                    new_structure: StructureV2 = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                        coords=coord_unpad,
                    )
                else:
                    new_structure: Structure = replace(
                        structure,
                        atoms=atoms,
                        residues=residues,
                        interfaces=interfaces,
                    )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Save the structure
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                # Get plddt's
                plddts = None
                if "plddt" in prediction:
                    plddts = prediction["plddt"][model_idx]

                # Create path name
                outname = f"{record.id}_model_{idx_to_rank[model_idx]}"

                # Save the structure
                if self.output_format == "pdb":
                    path = struct_dir / f"{outname}.pdb"
                    with path.open("w") as f:
                        f.write(
                            to_pdb(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                elif self.output_format == "mmcif":
                    path = struct_dir / f"{outname}.cif"
                    with path.open("w") as f:
                        f.write(
                            to_mmcif(new_structure, plddts=plddts, boltz2=self.boltz2)
                        )
                else:
                    path = struct_dir / f"{outname}.npz"
                    np.savez_compressed(path, **asdict(new_structure))

                if self.boltz2 and record.affinity and idx_to_rank[model_idx] == 0:
                    path = struct_dir / f"pre_affinity_{record.id}.npz"
                    np.savez_compressed(path, **asdict(new_structure))
                    np.array(atoms["coords"][:, None], dtype=Coords)

                # Save confidence summary
                if "plddt" in prediction:
                    path = (
                        struct_dir
                        / f"confidence_{record.id}_model_{idx_to_rank[model_idx]}.json"
                    )
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        confidence_summary_dict[key] = prediction[key][model_idx].item()
                    confidence_summary_dict["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][
                                model_idx
                            ].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save plddt
                    plddt = prediction["plddt"][model_idx]
                    path = (
                        struct_dir
                        / f"plddt_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, plddt=plddt.cpu().numpy())

                # Save pae
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    path = (
                        struct_dir
                        / f"pae_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pae=pae.cpu().numpy())

                # Save pde
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    path = (
                        struct_dir
                        / f"pde_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pde=pde.cpu().numpy())
                
            # Save embeddings
            if self.write_embeddings and "s" in prediction and "z" in prediction:
                s = prediction["s"].cpu().numpy()
                z = prediction["z"].cpu().numpy()

                path = (
                    struct_dir
                    / f"embeddings_{record.id}.npz"
                )
                np.savez_compressed(path, s=s, z=z)
        
        # Handle trajectory saving AFTER all structures are processed with final coordinates
        try:
            if "trajectory_coords" in prediction or "trajectory_denoised" in prediction:
                self._save_trajectories_with_final_coords(prediction, batch, records, coords, pad_masks, final_coords_for_trajectory)
        except Exception as e:
            print(f"[BoltzWriter] Trajectory saving failed: {e}")
            traceback.print_exc()

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
    
    def _save_trajectories_with_final_coords(
        self,
        prediction: dict[str, any],
        batch: dict[str, any],
        records: list[Record],
        coords: Tensor,
        pad_masks: Tensor,
        final_coords_for_trajectory: dict[str, dict[str, any]]
    ) -> None:
        """Save trajectory data using final processed coordinates to ensure consistency.
        
        Parameters
        ----------
        prediction : dict
            Prediction outputs containing trajectory data
        batch : dict
            Batch data with configuration
        records : list[Record]
            List of record objects
        coords : Tensor
            Final coordinates tensor
        pad_masks : Tensor
            Padding masks for coordinates
        final_coords_for_trajectory : dict
            Dictionary mapping record IDs to final coordinate data
        """
        trajectory_coords = prediction.get("trajectory_coords", [])  # Raw noisy coordinates
        trajectory_denoised = prediction.get("trajectory_denoised", [])  # Decoded predictions

        # Save trajectory for each record in the batch (usually just one)
        for i, (record, coord, pad_mask) in enumerate(zip(records, coords, pad_masks)):
            trajectory_writer = TrajectoryWriter(self.output_dir, record.id)
            
            # Get the final coordinates that were actually written to the output files
            final_data = final_coords_for_trajectory.get(record.id)
            if not final_data:
                continue
                
            final_coords = final_data['coords']  # These are the exact coordinates written to CIF
            
            # Extract trajectory for this batch item
            def extract_batch_trajectory(traj_list, final_coords_numpy):
                import torch
                batch_traj = []
                for step_coords in traj_list:
                    if step_coords.dim() > 2:  # Has batch dimension [batch, atoms, 3]
                        if i < step_coords.shape[0]:
                            # Extract this batch item and unpad properly
                            coords_batch = step_coords[i]
                            if pad_mask is not None and coords_batch.shape[0] == pad_mask.shape[0]:
                                mask_on_cpu = pad_mask.bool().cpu()
                                coords_on_cpu = coords_batch.cpu()
                                coords_unpadded = coords_on_cpu[mask_on_cpu]  # Keep as torch tensor
                            else:
                                coords_unpadded = coords_batch.cpu()  # Keep as torch tensor
                            batch_traj.append(coords_unpadded)
                        else:
                            break
                    else:  # No batch dimension [atoms, 3]
                        if pad_mask is not None and step_coords.shape[0] == pad_mask.shape[0]:
                            mask_on_cpu = pad_mask.bool().cpu()
                            coords_on_cpu = step_coords.cpu()
                            coords_unpadded = coords_on_cpu[mask_on_cpu]  # Keep as torch tensor
                        else:
                            coords_unpadded = step_coords.cpu()  # Keep as torch tensor
                        batch_traj.append(coords_unpadded)
                
                # Replace the last frame with the exact final coordinates
                if batch_traj and len(batch_traj) > 0:
                    # Convert final coordinates to torch tensor if needed
                    if isinstance(final_coords_numpy, torch.Tensor):
                        final_coords_tensor = final_coords_numpy.cpu()
                    else:
                        final_coords_tensor = torch.from_numpy(final_coords_numpy)
                    
                    # Ensure we use the exact same coordinates that were written to the output file
                    batch_traj[-1] = final_coords_tensor
                
                return batch_traj
            
            batch_trajectory_raw = extract_batch_trajectory(trajectory_coords, final_coords)
            batch_trajectory_denoised = extract_batch_trajectory(trajectory_denoised, final_coords)
            
            # Load structure info for trajectory writer
            structure_info = final_data['structure']
            
            metadata = {
                "record_id": record.id,
                "diffusion_steps_raw": len(batch_trajectory_raw),
                "diffusion_steps_denoised": len(batch_trajectory_denoised),
                "model_type": "boltz2" if self.boltz2 else "boltz1",
                "trajectory_atoms": batch_trajectory_raw[0].shape[0] if batch_trajectory_raw and len(batch_trajectory_raw) > 0 else 0,
                "final_coords_match": True  # Flag indicating final frame matches output
            }
            
            # Save both trajectories
            saved_files = {}
            
            if batch_trajectory_raw:
                saved_files.update(trajectory_writer.save_trajectory(
                    batch_trajectory_raw, pad_mask, metadata, structure_info, trajectory_type="raw"
                ))
            if batch_trajectory_denoised:
                saved_files.update(trajectory_writer.save_trajectory(
                    batch_trajectory_denoised, pad_mask, metadata, structure_info, trajectory_type="denoised"
                ))
            
                
            # Check for fixed chains and create unrotated trajectory if specified
            fixed_chains = []
            if "fixed_chains" in batch:
                fixed_chains_value = batch["fixed_chains"]
                if isinstance(fixed_chains_value, list):
                    if len(fixed_chains_value) > 0:
                        if isinstance(fixed_chains_value[0], list):
                            # Handle nested list case: [[B]] -> [B]
                            fixed_chains = fixed_chains_value[0]
                        else:
                            # Handle flat list case: [B] -> [B]
                            fixed_chains = fixed_chains_value
                elif hasattr(fixed_chains_value, 'item'):
                    # Handle tensor case
                    fixed_chains = fixed_chains_value.item() if hasattr(fixed_chains_value.item(), '__iter__') else []
                else:
                    fixed_chains = fixed_chains_value if hasattr(fixed_chains_value, '__iter__') else []
                    
            if fixed_chains and saved_files:
                # Find reference PDB file (input structure)
                reference_pdb = None
                possible_ref_paths = [
                    self.data_dir.parent / f"{record.id}.pdb",
                    Path.cwd() / f"{record.id}.pdb", 
                    Path.cwd() / "complex_rfdiffusion.pdb",
                    self.data_dir.parent / "complex_rfdiffusion.pdb"
                ]
                
                for ref_path in possible_ref_paths:
                    if ref_path.exists():
                        reference_pdb = ref_path
                        break
                
                # Create unrotated trajectory for both raw and denoised if they exist
                for trajectory_type in ["raw", "denoised"]:
                    if f"pdb_{trajectory_type}" in saved_files:
                        try:
                            create_unrotated_trajectory(
                                output_dir=self.output_dir,
                                record_id=record.id,
                                fixed_chains=fixed_chains,
                                reference_pdb=reference_pdb,
                                trajectory_type=trajectory_type
                            )
                        except Exception as e:
                            print(f"Error creating unrotated {trajectory_type} trajectory: {e}")
    

class BoltzAffinityWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        self.failed = 0
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return
        # Dump affinity summary
        affinity_summary = {}
        pred_affinity_value = prediction["affinity_pred_value"]
        pred_affinity_probability = prediction["affinity_probability_binary"]
        affinity_summary = {
            "affinity_pred_value": pred_affinity_value.item(),
            "affinity_probability_binary": pred_affinity_probability.item(),
        }
        if "affinity_pred_value1" in prediction:
            pred_affinity_value1 = prediction["affinity_pred_value1"]
            pred_affinity_probability1 = prediction["affinity_probability_binary1"]
            pred_affinity_value2 = prediction["affinity_pred_value2"]
            pred_affinity_probability2 = prediction["affinity_probability_binary2"]
            affinity_summary["affinity_pred_value1"] = pred_affinity_value1.item()
            affinity_summary["affinity_probability_binary1"] = (
                pred_affinity_probability1.item()
            )
            affinity_summary["affinity_pred_value2"] = pred_affinity_value2.item()
            affinity_summary["affinity_probability_binary2"] = (
                pred_affinity_probability2.item()
            )

        # Save the affinity summary
        struct_dir = self.output_dir / batch["record"][0].id
        struct_dir.mkdir(exist_ok=True)
        path = struct_dir / f"affinity_{batch['record'][0].id}.json"

        with path.open("w") as f:
            f.write(json.dumps(affinity_summary, indent=4))

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
