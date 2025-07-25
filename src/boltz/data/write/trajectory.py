"""
Trajectory writer for saving diffusion sampling trajectories.
Supports multiple formats for visualization with different tools.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple


class TrajectoryWriter:
    """Writer for diffusion trajectory data."""
    
    def __init__(self, output_dir: Path, record_id: str):
        """Initialize trajectory writer.
        
        Args:
            output_dir: Directory to save trajectory files
            record_id: Unique identifier for this trajectory
        """
        self.output_dir = Path(output_dir)
        self.record_id = record_id
        # Check if output_dir already ends with 'trajectories' to avoid nested directories
        if self.output_dir.name == "trajectories":
            self.trajectory_dir = self.output_dir
        else:
            self.trajectory_dir = self.output_dir / "trajectories"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        
    def save_trajectory(self, trajectory_coords: List[torch.Tensor], 
                       atom_mask: Optional[torch.Tensor] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       structure: Optional[Any] = None,
                       trajectory_type: str = "raw") -> Dict[str, Path]:
        """
        Save trajectory coordinates in multiple formats.
        
        Parameters
        ----------
        trajectory_coords : List[torch.Tensor]
            List of coordinate tensors, one per diffusion step
        atom_mask : Optional[torch.Tensor]
            Mask for valid atoms
        metadata : Optional[Dict[str, Any]]
            Additional metadata to save
        structure : Optional[Any]
            Structure object with atom and residue information
            
        Returns
        -------
        Dict[str, Path]
            Dictionary mapping format names to saved file paths
        """
        if not trajectory_coords:
            print("[TrajectoryWriter] No trajectory coordinates to save")
            return {}
            
        print(f"[TrajectoryWriter] Saving trajectory with {len(trajectory_coords)} frames")
        
        # Check if full structure data was provided in metadata
        if metadata and "full_structure_atoms" in metadata and metadata["full_structure_atoms"]:
            self._full_structure_atoms = metadata["full_structure_atoms"]
            print(f"[TrajectoryWriter] Received full structure data with {len(self._full_structure_atoms)} atoms")
        
        # Convert to numpy and handle batching
        coords_np = []
        for step_coords in trajectory_coords:
            # Handle batch dimension - take first sample if batched
            if len(step_coords.shape) == 3:  # [batch, atoms, 3]
                if hasattr(step_coords, 'numpy'):
                    coords = step_coords[0].numpy()  # Take first sample (PyTorch tensor)
                else:
                    coords = step_coords[0]  # Already numpy array
            else:  # [atoms, 3]
                if hasattr(step_coords, 'numpy'):
                    coords = step_coords.numpy()  # PyTorch tensor
                else:
                    coords = step_coords  # Already numpy array
            coords_np.append(coords)
        
        # Check for shape consistency before creating array
        shapes = [coords.shape for coords in coords_np]
        if len(set(shapes)) > 1:
            print(f"[TrajectoryWriter] Warning: Inconsistent frame shapes: {set(shapes)}")
            # Find the minimum common shape
            min_atoms = min(shape[0] for shape in shapes)
            print(f"[TrajectoryWriter] Truncating all frames to {min_atoms} atoms")
            coords_np = [coords[:min_atoms] for coords in coords_np]
        
        coords_array = np.array(coords_np)  # [frames, atoms, 3]
        
        # CRITICAL FIX: Expand trajectory frames to match full structure
        if hasattr(self, '_full_structure_atoms') and self._full_structure_atoms:
            full_atom_count = len(self._full_structure_atoms)
            traj_atom_count = coords_array.shape[1]
            
            if traj_atom_count < full_atom_count:
                print(f"[TrajectoryWriter] EXPANDING trajectory from {traj_atom_count} to {full_atom_count} atoms")
                
                # Create expanded coordinate array
                expanded_coords = np.zeros((coords_array.shape[0], full_atom_count, 3))
                
                # Copy trajectory atoms to first positions
                expanded_coords[:, :traj_atom_count, :] = coords_array
                
                # Fill remaining atoms with final structure coordinates
                final_coords = np.array([atom['coords'] for atom in self._full_structure_atoms])
                for frame_idx in range(coords_array.shape[0]):
                    expanded_coords[frame_idx, traj_atom_count:, :] = final_coords[traj_atom_count:]
                
                coords_array = expanded_coords
                print(f"[TrajectoryWriter] Expanded trajectory shape: {coords_array.shape}")
            else:
                print(f"[TrajectoryWriter] Trajectory already has full atom count: {traj_atom_count}")
        
        # Apply atom mask if provided - but check dimensions first
        if atom_mask is not None:
            mask_np = atom_mask.cpu().numpy().astype(bool)
            if len(mask_np.shape) == 1:  # [atoms]
                # Check if mask dimension matches trajectory dimension
                if mask_np.shape[0] == coords_array.shape[1]:
                    coords_array = coords_array[:, mask_np, :]
                else:
                    print(f"[TrajectoryWriter] Warning: Mask shape {mask_np.shape} doesn't match trajectory atom dimension {coords_array.shape[1]} - skipping mask")
            else:  # [batch, atoms] - take first batch
                mask_batch = mask_np[0]
                if mask_batch.shape[0] == coords_array.shape[1]:
                    coords_array = coords_array[:, mask_batch, :]
                else:
                    print(f"[TrajectoryWriter] Warning: Mask shape {mask_batch.shape} doesn't match trajectory atom dimension {coords_array.shape[1]} - skipping mask")
        
        saved_files = {}
        
        # Save as PDB trajectory (multi-model PDB)
        metadata_with_type = (metadata or {}).copy()
        metadata_with_type["trajectory_type"] = trajectory_type
        pdb_path = self.save_pdb_trajectory(coords_array, metadata_with_type, structure, trajectory_type)
        if pdb_path:
            saved_files[f"pdb_{trajectory_type}"] = pdb_path
            
        # Save metadata as JSON
        json_path = self.save_metadata(coords_array, metadata)
        if json_path:
            saved_files["json"] = json_path
            
        return saved_files
    
    
    def save_pdb_trajectory(self, coords_array: np.ndarray,
                           metadata: Optional[Dict[str, Any]] = None,
                           structure: Optional[Any] = None,
                           trajectory_type: str = "raw") -> Optional[Path]:
        """Save trajectory as multi-model PDB file with proper MODEL/END delimiters for molecular viewers."""
        try:
            pdb_path = self.trajectory_dir / f"{self.record_id}_trajectory_{trajectory_type}.pdb"
            num_frames, num_atoms, _ = coords_array.shape
            
            # Try to read the full atomic structure from the final prediction
            prediction_structure = self._read_prediction_structure()
            
            if prediction_structure:
                print(f"[TrajectoryWriter] Using full atomic structure from prediction with {len(prediction_structure)} atoms")
                print(f"[TrajectoryWriter] Trajectory has {num_atoms} atoms, prediction has {len(prediction_structure)} atoms")
                
                # Create mapping between trajectory atoms and prediction atoms
                # For multimer, trajectory typically contains all backbone atoms in order
                atom_mapping = self._create_atom_mapping(prediction_structure, num_atoms)
                
                with open(pdb_path, 'w') as f:
                    # Write header information
                    f.write(f"HEADER    DIFFUSION TRAJECTORY {trajectory_type.upper()}\n")
                    f.write(f"TITLE     BOLTZ DIFFUSION SAMPLING TRAJECTORY\n")
                    f.write(f"REMARK   1 TRAJECTORY TYPE: {trajectory_type.upper()}\n")
                    f.write(f"REMARK   1 NUMBER OF FRAMES: {num_frames}\n")
                    f.write(f"REMARK   1 TRAJECTORY ATOMS: {num_atoms}\n")
                    f.write(f"REMARK   1 TOTAL ATOMS IN STRUCTURE: {len(prediction_structure)}\n")
                    if num_atoms < len(prediction_structure):
                        f.write(f"REMARK   1 NOTE: TRAJECTORY CONTAINS SUBSET OF ATOMS DUE TO MODEL LIMITATIONS\n")
                        f.write(f"REMARK   1 SHOWING FIRST {num_atoms} ATOMS OF {len(prediction_structure)} TOTAL\n")
                    if metadata:
                        f.write(f"REMARK   1 METADATA: {metadata.get('trajectory_type', 'N/A')}\n")
                    f.write(f"REMARK   1 GENERATED BY BOLTZ TRAJECTORY WRITER\n")
                    
                    for frame_idx in range(num_frames):
                        # Standard multi-model PDB format with proper MODEL record
                        f.write(f"MODEL     {frame_idx + 1:4d}\n")
                        
                        # Write all atoms from prediction structure
                        for atom_idx, atom_record in enumerate(prediction_structure):
                            # Check if this atom has trajectory coordinates
                            if atom_idx in atom_mapping:
                                # Use trajectory coordinates
                                traj_idx = atom_mapping[atom_idx]
                                x, y, z = coords_array[frame_idx, traj_idx]
                            else:
                                # Atom not in trajectory - use original coordinates
                                # This can happen if trajectory has fewer atoms (e.g., no hydrogens)
                                x, y, z = atom_record['coords']
                            
                            # Write proper PDB ATOM record with correct formatting
                            f.write(f"ATOM  {atom_record['atom_num']:5d} {atom_record['atom_name']:>4s} "
                                   f"{atom_record['res_name']:>3s} {atom_record['chain_id']:>1s}{atom_record['res_num']:4d}    "
                                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{atom_record.get('b_factor', 20.00):6.2f}          "
                                   f"{atom_record['element']:>2s}\n")
                        
                        # Standard ENDMDL record to close the model
                        f.write("ENDMDL\n")
                    
                    # Final END record to close the file
                    f.write("END\n")
                
                print(f"[TrajectoryWriter] Saved multi-model PDB trajectory with {num_frames} frames: {pdb_path}")
                return pdb_path
            
            else:
                # Fallback to previous method if no prediction structure available
                print(f"[TrajectoryWriter] No prediction structure found, using simplified backbone")
                return self._save_simple_trajectory(pdb_path, coords_array, structure)
            
        except Exception as e:
            print(f"[TrajectoryWriter] Failed to save PDB trajectory: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _save_simple_trajectory(self, pdb_path: Path, coords_array: np.ndarray, structure: Optional[Any] = None) -> Optional[Path]:
        """Fallback method for simple trajectory without full atomic structure."""
        try:
            num_frames, num_atoms, _ = coords_array.shape
            
            # Try to read atom info from original input PDB
            atom_info = self._read_input_pdb_structure(num_atoms)
            if not atom_info:
                atom_info = self._extract_atom_info(structure, num_atoms) if structure else None
            
            with open(pdb_path, 'w') as f:
                # Write header information for simple trajectory
                f.write(f"HEADER    BACKBONE DIFFUSION TRAJECTORY\n")
                f.write(f"TITLE     BOLTZ BACKBONE TRAJECTORY (SUBSET OF ATOMS)\n")
                f.write(f"REMARK   1 BACKBONE TRAJECTORY (PREDICTION STRUCTURE NOT FOUND)\n")
                f.write(f"REMARK   1 NUMBER OF FRAMES: {num_frames}\n")
                f.write(f"REMARK   1 ATOMS PER FRAME: {num_atoms}\n")
                f.write(f"REMARK   1 TRAJECTORY CONTAINS BACKBONE ATOMS ONLY\n")
                f.write(f"REMARK   1 GENERATED BY BOLTZ TRAJECTORY WRITER\n")
                
                for frame_idx in range(num_frames):
                    f.write(f"MODEL     {frame_idx + 1:4d}\n")
                    
                    atom_counter = 1
                    
                    if atom_info:
                        for atom_idx in range(min(num_atoms, len(atom_info))):
                            atom_name, res_name, chain_id, res_num, element = atom_info[atom_idx]
                            x, y, z = coords_array[frame_idx, atom_idx]
                            
                            f.write(f"ATOM  {atom_counter:5d} {atom_name:>4s} {res_name:>3s} {chain_id:>1s}{res_num:4d}    "
                                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n")
                            atom_counter += 1
                    else:
                        # The trajectory contains ALL atoms, not just backbone
                        # Since we couldn't read the structure, create a basic representation
                        print(f"[TrajectoryWriter] WARNING: Creating basic atom representation for {num_atoms} atoms")
                        
                        # Estimate atoms per residue based on total atom count
                        # Typical protein has ~8-14 heavy atoms per residue
                        estimated_residues = num_atoms // 10  # Rough estimate
                        
                        for atom_idx in range(num_atoms):
                            x, y, z = coords_array[frame_idx, atom_idx]
                            
                            # Simple sequential numbering
                            res_num = (atom_idx // 10) + 1  # Assuming ~10 atoms per residue
                            
                            # Determine chain based on residue number
                            if estimated_residues > 100:  # Likely multimer
                                chain_id = chr(ord('A') + (res_num - 1) // (estimated_residues // 2))
                                if chain_id > 'B':
                                    chain_id = 'B'
                                adjusted_res_num = res_num if chain_id == 'A' else res_num - (estimated_residues // 2)
                            else:
                                chain_id = 'A'
                                adjusted_res_num = res_num
                            
                            # Create generic atom names
                            atom_in_res = atom_idx % 10
                            if atom_in_res == 0:
                                atom_name = 'N'
                                element = 'N'
                            elif atom_in_res == 1:
                                atom_name = 'CA'
                                element = 'C'
                            elif atom_in_res == 2:
                                atom_name = 'C'
                                element = 'C'
                            elif atom_in_res == 3:
                                atom_name = 'O'
                                element = 'O'
                            else:
                                atom_name = f'C{atom_in_res-3}'
                                element = 'C'
                            
                            f.write(f"ATOM  {atom_counter:5d} {atom_name:>4s} ALA {chain_id:>1s}{adjusted_res_num:4d}    "
                                   f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n")
                            atom_counter += 1
                    
                    f.write("ENDMDL\n")
                
                f.write("END\n")
            
            print(f"[TrajectoryWriter] Saved backbone trajectory with {num_frames} frames: {pdb_path}")
            return pdb_path
            
        except Exception as e:
            print(f"[TrajectoryWriter] Failed to save simple trajectory: {e}")
            return None
    
    def _read_input_pdb_structure(self, num_atoms: int) -> Optional[List[Tuple[str, str, str, int, str]]]:
        """Try to read atom info from the original input PDB file."""
        try:
            # Look for input PDB files
            input_files = [
                Path.cwd() / f"{self.record_id}.pdb",
                Path.cwd() / "complex_rfdiffusion.pdb",  # Common name
                self.output_dir.parent.parent / f"{self.record_id}.pdb",
                self.output_dir.parent.parent / "complex_rfdiffusion.pdb"
            ]
            
            input_file = None
            for potential_path in input_files:
                if potential_path.exists():
                    input_file = potential_path
                    print(f"[TrajectoryWriter] Found input PDB: {potential_path}")
                    break
            
            if not input_file:
                return None
                
            atom_info = []
            backbone_atoms = ['N', 'CA', 'C', 'O']
            
            with open(input_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM  '):
                        atom_name = line[12:16].strip()
                        if atom_name in backbone_atoms:
                            res_name = line[17:20].strip()
                            chain_id = line[21].strip() or 'A'
                            res_num = int(line[22:26].strip())
                            element = atom_name[0] if atom_name else 'C'
                            atom_info.append((atom_name, res_name, chain_id, res_num, element))
                            
                            if len(atom_info) >= num_atoms:
                                break
            
            print(f"[TrajectoryWriter] Read {len(atom_info)} backbone atoms from input PDB")
            return atom_info if atom_info else None
            
        except Exception as e:
            print(f"[TrajectoryWriter] Failed to read input PDB structure: {e}")
            return None
    
    def _read_prediction_structure(self) -> Optional[List[Dict]]:
        """Read the full atomic structure from the prediction CIF/PDB file."""
        try:
            # PRIORITY: Check if full structure was provided by writer
            if hasattr(self, '_full_structure_atoms') and self._full_structure_atoms:
                print(f"[TrajectoryWriter] Using full structure data provided by writer ({len(self._full_structure_atoms)} atoms)")
                return self._full_structure_atoms
            # Look for prediction structure files - comprehensive search for multimer support
            prediction_files = [
                # Most common path: predictions/record_id/record_id_model_0.cif
                self.output_dir.parent.parent / self.record_id / f"{self.record_id}_model_0.cif",
                self.output_dir.parent.parent / self.record_id / f"{self.record_id}_model_0.pdb",
                # Standard output structure: output_dir/record_id/record_id_model_0.cif
                self.output_dir.parent / self.record_id / f"{self.record_id}_model_0.cif",
                self.output_dir.parent / self.record_id / f"{self.record_id}_model_0.pdb",
                # Alternative: output_dir/predictions/record_id/record_id_model_0.cif
                self.output_dir.parent / "predictions" / self.record_id / f"{self.record_id}_model_0.cif",
                self.output_dir.parent / "predictions" / self.record_id / f"{self.record_id}_model_0.pdb",
                # Search in current working directory
                Path.cwd() / "boltz_results_complex_rfdiffusion" / "predictions" / self.record_id / f"{self.record_id}_model_0.cif",
                Path.cwd() / "boltz_results_complex_rfdiffusion" / "predictions" / self.record_id / f"{self.record_id}_model_0.pdb",
                # Generic patterns
                Path.cwd() / f"boltz_results_{self.record_id}" / "predictions" / self.record_id / f"{self.record_id}_model_0.cif",
                Path.cwd() / f"boltz_results_{self.record_id}" / "predictions" / self.record_id / f"{self.record_id}_model_0.pdb",
                # Relative to trajectory directory
                self.trajectory_dir.parent.parent / "predictions" / self.record_id / f"{self.record_id}_model_0.cif",
                self.trajectory_dir.parent.parent / "predictions" / self.record_id / f"{self.record_id}_model_0.pdb",
                # Legacy paths
                self.output_dir.parent / "pdb" / f"{self.record_id}_model_0.cif",
                self.output_dir.parent / "pdb" / f"{self.record_id}_model_0.pdb"
            ]
            
            prediction_file = None
            print(f"[TrajectoryWriter] Searching for prediction files for record_id: {self.record_id}")
            print(f"[TrajectoryWriter] Output dir: {self.output_dir}")
            
            for potential_path in prediction_files:
                print(f"[TrajectoryWriter] Checking: {potential_path}")
                if potential_path.exists():
                    prediction_file = potential_path
                    print(f"[TrajectoryWriter] Found prediction file: {potential_path}")
                    break
            
            if not prediction_file:
                print(f"[TrajectoryWriter] No prediction structure file found")
                print(f"[TrajectoryWriter] Searched {len(prediction_files)} paths")
                return None
            
            print(f"[TrajectoryWriter] Reading prediction structure from {prediction_file}")
            
            atoms = []
            with open(prediction_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM  ') or (line.startswith('ATOM ') and prediction_file.suffix == '.cif'):
                        if prediction_file.suffix == '.cif':
                            # Parse mmCIF format: ATOM atom_id element atom_name alt_loc res_name seq_id auth_seq_id ins_code chain_id x y z occupancy entity_id auth_chain_id auth_res_name b_factor model_num
                            parts = line.split()
                            if len(parts) >= 19:
                                atom_record = {
                                    'atom_num': int(parts[1]),
                                    'element': parts[2],
                                    'atom_name': parts[3],
                                    'res_name': parts[5],
                                    'res_num': int(parts[6]),  # seq_id, not auth_seq_id
                                    'chain_id': parts[9],
                                    'coords': [float(parts[10]), float(parts[11]), float(parts[12])],
                                    'b_factor': float(parts[17])
                                }
                                atoms.append(atom_record)
                        else:
                            # Parse PDB format
                            atom_record = {
                                'atom_num': int(line[6:11].strip()),
                                'atom_name': line[12:16].strip(),
                                'res_name': line[17:20].strip(),
                                'chain_id': line[21].strip() or 'A',
                                'res_num': int(line[22:26].strip()),
                                'coords': [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                                'element': line[76:78].strip() or line[12:16].strip()[0],
                                'b_factor': float(line[60:66]) if line[60:66].strip() else 20.0
                            }
                            atoms.append(atom_record)
            
            print(f"[TrajectoryWriter] Successfully read {len(atoms)} atoms from prediction structure")
            if atoms:
                # Debug: show chain distribution
                chain_counts = {}
                for atom in atoms:
                    chain_id = atom['chain_id']
                    chain_counts[chain_id] = chain_counts.get(chain_id, 0) + 1
                print(f"[TrajectoryWriter] Chain distribution: {chain_counts}")
            return atoms if atoms else None
            
        except Exception as e:
            print(f"[TrajectoryWriter] Failed to read prediction structure: {e}")
            return None
    
    def _create_atom_mapping(self, prediction_structure: List[Dict], num_trajectory_atoms: int) -> Dict[int, int]:
        """Create mapping between prediction atoms and trajectory atoms.
        
        IMPORTANT: The trajectory contains ALL atoms from the structure, not just backbone atoms.
        The atoms are in the same order as they appear in the tokenized structure data.
        """
        mapping = {}
        
        print(f"[TrajectoryWriter] Creating atom mapping for {len(prediction_structure)} prediction atoms to {num_trajectory_atoms} trajectory atoms")
        
        # First, check if we have a direct 1:1 mapping (same number of atoms)
        if len(prediction_structure) == num_trajectory_atoms:
            print(f"[TrajectoryWriter] Direct 1:1 mapping - trajectory contains all atoms")
            # Simple direct mapping - atoms are in the same order
            for i in range(num_trajectory_atoms):
                mapping[i] = i
            return mapping
        
        # If not 1:1, we need to figure out which atoms are in the trajectory
        # The trajectory might contain a subset of atoms (e.g., only heavy atoms, no hydrogens)
        
        # Group atoms by residue for analysis
        residue_groups = {}
        for pred_idx, atom_record in enumerate(prediction_structure):
            chain_id = atom_record['chain_id']
            res_num = atom_record['res_num']
            key = (chain_id, res_num)
            if key not in residue_groups:
                residue_groups[key] = []
            residue_groups[key].append((pred_idx, atom_record))
        
        print(f"[TrajectoryWriter] Found {len(residue_groups)} residues, {len(prediction_structure)} total atoms")
        print(f"[TrajectoryWriter] Trajectory has {num_trajectory_atoms} atoms")
        
        # Calculate atoms per residue to understand the pattern
        atoms_per_residue = num_trajectory_atoms / len(residue_groups) if len(residue_groups) > 0 else 0
        print(f"[TrajectoryWriter] Average atoms per residue in trajectory: {atoms_per_residue:.1f}")
        
        # Try different mapping strategies based on the atom count pattern
        
        # Strategy 1: If trajectory has fewer atoms, it might exclude hydrogens
        if num_trajectory_atoms < len(prediction_structure):
            print(f"[TrajectoryWriter] Trajectory has fewer atoms - likely excluding hydrogens or some atoms")
            
            trajectory_idx = 0
            # Map non-hydrogen atoms in order
            for pred_idx, atom_record in enumerate(prediction_structure):
                if trajectory_idx >= num_trajectory_atoms:
                    break
                    
                atom_name = atom_record['atom_name'].strip()
                element = atom_record.get('element', atom_name[0]).strip()
                
                # Skip hydrogens if they're likely excluded
                if element == 'H' and num_trajectory_atoms < len(prediction_structure) * 0.8:
                    continue
                    
                mapping[pred_idx] = trajectory_idx
                trajectory_idx += 1
            
            # If we didn't map enough atoms, try mapping all atoms sequentially
            if len(mapping) < num_trajectory_atoms:
                print(f"[TrajectoryWriter] First strategy only mapped {len(mapping)} atoms, trying sequential mapping")
                mapping = {}
                for i in range(min(len(prediction_structure), num_trajectory_atoms)):
                    mapping[i] = i
        
        # Strategy 2: If trajectory has more atoms than expected, handle padding or special cases
        elif num_trajectory_atoms > len(prediction_structure):
            print(f"[TrajectoryWriter] Warning: Trajectory has MORE atoms than prediction structure")
            # Map all prediction atoms to the first trajectory atoms
            for i in range(len(prediction_structure)):
                mapping[i] = i
        
        # Validate the mapping
        if len(mapping) == 0:
            print(f"[TrajectoryWriter] WARNING: No atoms mapped! Using fallback sequential mapping")
            for i in range(min(len(prediction_structure), num_trajectory_atoms)):
                mapping[i] = i
        
        print(f"[TrajectoryWriter] Final mapping: {len(mapping)} atoms mapped")
        
        # Debug: Show atom distribution in mapping
        if len(mapping) > 0:
            atom_types = {}
            for pred_idx in mapping:
                atom_name = prediction_structure[pred_idx]['atom_name'].strip()
                atom_types[atom_name] = atom_types.get(atom_name, 0) + 1
            
            # Show most common atom types
            sorted_types = sorted(atom_types.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"[TrajectoryWriter] Top atom types in mapping: {sorted_types}")
            
            # Check chain distribution
            chain_counts = {}
            for pred_idx in mapping:
                chain_id = prediction_structure[pred_idx]['chain_id']
                chain_counts[chain_id] = chain_counts.get(chain_id, 0) + 1
            print(f"[TrajectoryWriter] Atoms per chain: {chain_counts}")
        
        return mapping
            
    def _read_pdb_atom_info(self, num_atoms: int) -> Optional[List[Tuple[str, str, str, int, str]]]:
        """Read atom information directly from original PDB file."""
        try:
            # Look for original PDB file in multiple possible locations
            pdb_file = None
            search_paths = [
                # In the parent directory
                self.output_dir.parent / f"{self.record_id}.pdb",
                self.output_dir.parent / f"{self.record_id}.cif",
                # In the current working directory
                Path(f"{self.record_id}.pdb"),
                Path(f"{self.record_id}.cif"),
                # Try common names
                self.output_dir.parent / "pdb.pdb", 
                self.output_dir.parent / "pdb.cif",
                Path("pdb.pdb"),
                Path("pdb.cif"),
                # Additional search paths - go up more directories  
                self.output_dir.parent.parent / f"{self.record_id}.pdb",
                self.output_dir.parent.parent / f"{self.record_id}.cif",
                self.output_dir.parent.parent / "pdb.pdb",
                self.output_dir.parent.parent / "pdb.cif",
                # And one more level up
                self.output_dir.parent.parent.parent / f"{self.record_id}.pdb",
                self.output_dir.parent.parent.parent / f"{self.record_id}.cif", 
                self.output_dir.parent.parent.parent / "pdb.pdb",
                self.output_dir.parent.parent.parent / "pdb.cif"
            ]
            
            for potential_path in search_paths:
                if potential_path.exists():
                    pdb_file = potential_path
                    break
            
            if not pdb_file:
                print(f"[TrajectoryWriter] Original PDB file not found for {self.record_id}")
                print(f"[TrajectoryWriter] Searched paths: {[str(p) for p in search_paths]}")
                return None
                
            print(f"[TrajectoryWriter] Reading atom info from {pdb_file}")
            atom_info = []
            
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM  '):
                        # Parse PDB ATOM record
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        chain_id = line[21].strip() or 'A'
                        res_num = int(line[22:26].strip())
                        
                        # Infer element from atom name
                        element = atom_name[0] if atom_name else 'C'
                        if element in ['N', 'O', 'S', 'P']:
                            pass  # Keep as is
                        else:
                            element = 'C'  # Default to carbon
                        
                        atom_info.append((atom_name, res_name, chain_id, res_num, element))
                        
                        if len(atom_info) >= num_atoms:
                            break
            
            if len(atom_info) < num_atoms:
                print(f"[TrajectoryWriter] Warning: Found {len(atom_info)} atoms in PDB but expected {num_atoms}")
                # Pad with default atoms
                while len(atom_info) < num_atoms:
                    atom_info.append(("CA", "UNK", "A", len(atom_info) + 1, "C"))
            
            print(f"[TrajectoryWriter] Successfully read {len(atom_info)} atoms from PDB")
            return atom_info
            
        except Exception as e:
            print(f"[TrajectoryWriter] Failed to read PDB atom info: {e}")
            return None
    
    def save_metadata(self, coords_array: np.ndarray,
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """Save trajectory metadata as JSON."""
        try:
            json_path = self.trajectory_dir / f"{self.record_id}_trajectory_metadata.json"
            
            trajectory_info = {
                "record_id": self.record_id,
                "num_frames": int(coords_array.shape[0]),
                "num_atoms": int(coords_array.shape[1]),
                "coordinates_shape": list(coords_array.shape),
                "coordinate_units": "Angstroms",
                "source": "Boltz diffusion sampling",
                "metadata": metadata or {}
            }
            
            with open(json_path, 'w') as f:
                json.dump(trajectory_info, f, indent=2)
            
            print(f"[TrajectoryWriter] Saved metadata: {json_path}")
            return json_path
            
        except Exception as e:
            print(f"[TrajectoryWriter] Failed to save metadata: {e}")
            return None
    
 

    def _extract_atom_info(self, structure: Any, num_atoms: int) -> List[Tuple[str, str, str, int, str]]:
        """
        Extract atom information from structure object for PDB writing.
        
        Parameters
        ----------
        structure : Any
            Structure object with atoms, residues, and chains
        num_atoms : int
            Number of atoms in the trajectory
            
        Returns
        -------
        List[Tuple[str, str, str, int, str]]
            List of tuples containing (atom_name, res_name, chain_id, res_num, element)
        """
        try:
            if not structure:
                # No structure available, use defaults
                return [("CA", "UNK", "A", i + 1, "C") for i in range(num_atoms)]
            
            # Try to extract real atom info, fall back to reading from prediction structure
            try:
                print(f"[TrajectoryWriter] Attempting to extract atom info from structure")
                
                # Try the complex extraction but with better error handling
                if hasattr(structure, 'atoms') and hasattr(structure, 'residues'):
                    atoms = structure.atoms
                    residues = structure.residues
                    chains = getattr(structure, 'chains', None)
                    
                    print(f"[TrajectoryWriter] Found {len(atoms)} atoms, {len(residues)} residues")
                    
                    # Debug: check what's in the atoms and residues
                    if len(atoms) > 0:
                        print(f"[TrajectoryWriter] Sample atom keys: {list(atoms[0].keys()) if hasattr(atoms[0], 'keys') else 'no keys'}")
                    if len(residues) > 0:  
                        print(f"[TrajectoryWriter] Sample residue keys: {list(residues[0].keys()) if hasattr(residues[0], 'keys') else 'no keys'}")
                    
                    atom_info = []
                    for atom_idx in range(min(num_atoms, len(atoms))):
                        try:
                            atom = atoms[atom_idx]
                            
                            # Extract atom name safely
                            atom_name = "CA"
                            if hasattr(atom, 'get') and 'name' in atom:
                                name_val = atom['name']
                                if hasattr(name_val, 'item'):
                                    atom_name = str(name_val.item()).strip() or "CA"
                                else:
                                    atom_name = str(name_val).strip() or "CA"
                            
                            # Extract element safely  
                            element = "C"
                            if hasattr(atom, 'get') and 'element' in atom:
                                elem_val = atom['element']
                                if hasattr(elem_val, 'item'):
                                    elem_idx = int(elem_val.item())
                                else:
                                    elem_idx = int(elem_val)
                                
                                # Simple element mapping
                                elem_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 16: 'S', 15: 'P'}
                                element = elem_map.get(elem_idx, 'C')
                            
                            # Try to extract residue info from structure
                            res_name = "GLY"  # Default fallback
                            chain_id = "A"
                            res_num = (atom_idx // 4) + 1
                            
                            # Try to find which residue this atom belongs to
                            try:
                                for res_idx, res in enumerate(residues):
                                    atom_start = int(res.get('atom_idx', 0))
                                    atom_num = int(res.get('atom_num', 4))
                                    atom_end = atom_start + atom_num
                                    
                                    if atom_start <= atom_idx < atom_end:
                                        if 'name' in res:
                                            res_name_raw = res['name']
                                            if hasattr(res_name_raw, 'item'):
                                                res_name = str(res_name_raw.item()).strip()
                                            else:
                                                res_name = str(res_name_raw).strip()
                                            if len(res_name) == 3:  # Valid residue name
                                                pass  # Keep it
                                            else:
                                                res_name = "ALA"  # Better fallback than GLY
                                        res_num = int(res.get('res_idx', res_idx)) + 1
                                        break
                            except Exception as e:
                                pass  # Keep defaults
                            
                            atom_info.append((atom_name, res_name, chain_id, res_num, element))
                            
                        except Exception as e:
                            # Safe fallback for individual atoms
                            atom_info.append(("CA", "GLY", "A", atom_idx + 1, "C"))
                    
                    return atom_info
                    
            except Exception as e:
                print(f"[TrajectoryWriter] Complex extraction failed: {e}")
            
            # Try to read from original PDB file before falling back to generic
            pdb_atom_info = self._read_pdb_atom_info(num_atoms)
            if pdb_atom_info:
                print(f"[TrajectoryWriter] Using atom info from original PDB file")
                return pdb_atom_info
            
            # Final fallback - use generic but varied atom names
            print(f"[TrajectoryWriter] Using generic atom names as fallback")
            generic_atoms = []
            for i in range(num_atoms):
                atom_types = ["CA", "CB", "N", "O", "C"]
                elements = ["C", "C", "N", "O", "C"]
                idx = i % len(atom_types)
                res_num = (i // len(atom_types)) + 1
                
                generic_atoms.append((
                    atom_types[idx],
                    "ALA",  # Use ALA instead of GLY
                    "A", 
                    res_num, 
                    elements[idx]
                ))
            return generic_atoms
            
        except Exception as e:
            print(f"[TrajectoryWriter] Failed to extract atom info: {e}")
            import traceback
            traceback.print_exc()
            # Return default data
            return [("CA", "UNK", "A", i + 1, "C") for i in range(num_atoms)]
    
 