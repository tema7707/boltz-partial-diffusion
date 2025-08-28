# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import nn
from torch.nn import Module

import boltz.model.layers.initialize as init
from boltz.data import const
from boltz.model.loss.diffusionv2 import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encodersv2 import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    SingleConditioning,
)
from boltz.model.modules.transformersv2 import (
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    compute_random_augmentation,
    default,
    log,
)
from boltz.model.potentials.potentials import get_potentials


class DiffusionModule(Module):

    def __init__(
        self,
        token_s: int,
        atom_s: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        transformer_post_ln: bool = False,
    ) -> None:
        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.activation_checkpointing = activation_checkpointing

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            token_s=token_s,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
            transformer_post_layer_norm=transformer_post_ln,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            # post_layer_norm=transformer_post_ln,
        )

        self.a_norm = nn.LayerNorm(
            2 * token_s
        )  # if not transformer_post_ln else nn.Identity()

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
            # transformer_post_layer_norm=transformer_post_ln,
        )

    def forward(
        self,
        s_inputs,  # Float['b n ts']
        s_trunk,  # Float['b n ts']
        r_noisy,  # Float['bm m 3']
        times,  # Float['bm 1 1']
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        if self.activation_checkpointing and self.training:
            s, normed_fourier = torch.utils.checkpoint.checkpoint(
                self.single_conditioner,
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )
        else:
            s, normed_fourier = self.single_conditioner(
                times,
                s_trunk.repeat_interleave(multiplicity, 0),
                s_inputs.repeat_interleave(multiplicity, 0),
            )

        a, q_skip, c_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            q=diffusion_conditioning["q"].float(),
            c=diffusion_conditioning["c"].float(),
            atom_enc_bias=diffusion_conditioning["atom_enc_bias"].float(),
            to_keys=diffusion_conditioning["to_keys"],
            r=r_noisy,  # Float['b m 3'],
            multiplicity=multiplicity,
        )

        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            bias=diffusion_conditioning[
                "token_trans_bias"
            ].float(),  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
        )
        a = self.a_norm(a)

        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            atom_dec_bias=diffusion_conditioning["atom_dec_bias"].float(),
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
        )

        return r_update


class AtomDiffusion(Module):
    def __init__(
        self,
        score_model_args,
        num_sampling_steps: int = 5,  # number of sampling steps
        sigma_min: float = 0.0004,  # min noise level
        sigma_max: float = 160.0,  # max noise level
        sigma_data: float = 16.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.5,  # standard deviation of log-normal distribution from which noise is drawn for training
        gamma_0: float = 0.8,
        gamma_min: float = 1.0,
        noise_scale: float = 1.003,
        step_scale: float = 1.5,
        step_scale_random: list = None,
        coordinate_augmentation: bool = True,
        coordinate_augmentation_inference=None,
        compile_score: bool = False,
        alignment_reverse_diff: bool = False,
        synchronize_sigmas: bool = False,
    ):
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.step_scale_random = step_scale_random
        self.coordinate_augmentation = coordinate_augmentation
        self.coordinate_augmentation_inference = (
            coordinate_augmentation_inference
            if coordinate_augmentation_inference is not None
            else coordinate_augmentation
        )
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas

        self.token_s = score_model_args["token_s"]
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,  #: Float['b m 3'],
        sigma,  #: Float['b'] | Float[' '] | float,
        network_condition_kwargs: dict,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        r_update = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * r_update
        )
        return denoised_coords

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        max_parallel_samples=None,
        steering_args=None,
        **network_condition_kwargs,
    ):
        if steering_args is not None and (
            steering_args["fk_steering"]
            or steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ):
            potentials = get_potentials(steering_args, boltz2=True)

        if steering_args["fk_steering"]:
            multiplicity = multiplicity * steering_args["num_particles"]
            energy_traj = torch.empty((multiplicity, 0), device=self.device)
            resample_weights = torch.ones(multiplicity, device=self.device).reshape(
                -1, steering_args["num_particles"]
            )
        if (
            steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ):
            scaled_guidance_update = torch.zeros(
                (multiplicity, *atom_mask.shape[1:], 3),
                dtype=torch.float32,
                device=self.device,
            )
        if max_parallel_samples is None:
            max_parallel_samples = multiplicity

        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        
        feats = network_condition_kwargs.get("feats", {})
        partial_diffusion_fraction = feats.get("partial_diffusion_fraction", 0.0)
        if hasattr(partial_diffusion_fraction, 'item'):
            partial_diffusion_fraction = partial_diffusion_fraction.item()
        original_num_steps = num_sampling_steps
        partial_diffusion_skip_steps = 0
        
        if partial_diffusion_fraction > 0.0 and "initial_coords" in feats:
            partial_diffusion_skip_steps = int(original_num_steps * (1.0 - partial_diffusion_fraction))
            actual_steps = original_num_steps - partial_diffusion_skip_steps
            actual_steps = max(1, actual_steps)
            num_sampling_steps = actual_steps
        
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        if partial_diffusion_skip_steps > 0:
            full_sigmas = self.sample_schedule(original_num_steps)
            sigmas = full_sigmas[partial_diffusion_skip_steps:partial_diffusion_skip_steps + num_sampling_steps + 1]
        else:
            sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))
        if self.training and self.step_scale_random is not None:
            step_scale = np.random.choice(self.step_scale_random)
        else:
            step_scale = self.step_scale

        save_trajectory = network_condition_kwargs.get("feats", {}).get("save_trajectory", False)
        trajectory_decoded_only = network_condition_kwargs.get("feats", {}).get("trajectory_decoded_only", False)
        trajectory_coords = []
        trajectory_denoised = []

        init_sigma = sigmas[0]
        feats = network_condition_kwargs.get("feats", {})
        if "initial_coords" in feats and feats["initial_coords"] is not None:
            initial_coords = feats["initial_coords"]
            
            if len(initial_coords.shape) == 2:
                initial_coords = initial_coords.unsqueeze(0)
            
            initial_coords = initial_coords.to(device=self.device, dtype=torch.float32)
            
            # CRITICAL FIX: Apply canonical space transformation to initial coordinates
            # Center the coordinates like Boltz does during training (center_random_augmentation)
            initial_coords_mean = initial_coords.mean(dim=-2, keepdims=True)
            initial_coords = initial_coords - initial_coords_mean
            
            initial_coords = initial_coords.repeat_interleave(multiplicity, 0)
            
            if initial_coords.shape != shape:
                expected_atoms = shape[1]
                actual_atoms = initial_coords.shape[-2]
                
                if actual_atoms < expected_atoms:
                    padding_needed = expected_atoms - actual_atoms
                    padding = torch.zeros((initial_coords.shape[0], padding_needed, 3), 
                                        device=initial_coords.device, dtype=initial_coords.dtype)
                    initial_coords = torch.cat([initial_coords, padding], dim=1)
                elif actual_atoms > expected_atoms:
                    initial_coords = initial_coords[:, :expected_atoms, :]
            
            if partial_diffusion_fraction > 0.0:
                starting_sigma = init_sigma
                noise = torch.randn_like(initial_coords)
                atom_coords = initial_coords + starting_sigma * noise
            else:
                noise = torch.randn_like(initial_coords)
                atom_coords = initial_coords + 0.01 * init_sigma * noise
                
        else:
            atom_coords = init_sigma * torch.randn(shape, device=self.device)
        token_repr = None
        atom_coords_denoised = None

        fixed_chains_data = None
        if "fixed_chains" in feats and "initial_coords" in feats and len(feats["fixed_chains"]) > 0:
            try:
                fixed_asym_ids = feats["fixed_chains"]  # These are already asym_id indices (tensors)
                initial_coords = feats["initial_coords"]
                
                
                # Convert to tensor if needed
                if not isinstance(fixed_asym_ids, torch.Tensor):
                    fixed_asym_ids = torch.tensor(fixed_asym_ids, dtype=torch.long, device=atom_coords.device)
                else:
                    fixed_asym_ids = fixed_asym_ids.to(device=atom_coords.device)
                
                if not isinstance(initial_coords, torch.Tensor):
                    initial_coords = torch.tensor(initial_coords, dtype=atom_coords.dtype, device=atom_coords.device)
                else:
                    initial_coords = initial_coords.to(device=atom_coords.device, dtype=atom_coords.dtype)
                
                asym_id = feats["asym_id"]
                if not isinstance(asym_id, torch.Tensor):
                    asym_id = torch.tensor(asym_id, dtype=torch.long, device=atom_coords.device)
                else:
                    asym_id = asym_id.to(device=atom_coords.device)
                
                # Convert token-level asym_id to atom-level asym_id
                atom_to_token = feats["atom_to_token"]
                asym_id_atom = torch.bmm(atom_to_token.float(), asym_id.unsqueeze(-1).float()).squeeze(-1).long()
                
                # DEBUG: Print asym_id information
                # DEBUG: Token-level asym_id: {asym_id}")
                # DEBUG: Atom-level asym_id unique values: {torch.unique(asym_id_atom)}")
                # DEBUG: Fixed chain IDs requested: {fixed_asym_ids}")
                
                fixed_mask = torch.zeros_like(asym_id_atom, dtype=torch.bool, device=atom_coords.device)
                for fixed_asym_id in fixed_asym_ids:
                    mask_for_this_id = (asym_id_atom == fixed_asym_id)
                    num_atoms_for_chain = mask_for_this_id.sum().item()
                    # DEBUG: Chain {fixed_asym_id}: {num_atoms_for_chain} atoms will be fixed")
                    fixed_mask |= mask_for_this_id
                
                if len(fixed_mask.shape) == 1:
                    fixed_mask = fixed_mask.unsqueeze(0).unsqueeze(-1)
                elif len(fixed_mask.shape) == 2:
                    fixed_mask = fixed_mask.unsqueeze(-1)
                
                if len(initial_coords.shape) == 2:
                    initial_coords = initial_coords.unsqueeze(0)
                
                if initial_coords.shape[0] != atom_coords.shape[0]:
                    initial_coords = initial_coords.repeat(atom_coords.shape[0], 1, 1)
                
                if initial_coords.shape != atom_coords.shape:
                    if initial_coords.shape[1] < atom_coords.shape[1]:
                        padding_needed = atom_coords.shape[1] - initial_coords.shape[1]
                        if initial_coords.shape[1] > 0:
                            last_coord = initial_coords[:, -1:, :].expand(-1, padding_needed, -1)
                            initial_coords = torch.cat([initial_coords, last_coord], dim=1)
                        else:
                            padding = torch.zeros((initial_coords.shape[0], padding_needed, 3), 
                                                device=initial_coords.device, dtype=initial_coords.dtype)
                            initial_coords = torch.cat([initial_coords, padding], dim=1)
                    elif initial_coords.shape[1] > atom_coords.shape[1]:
                        initial_coords = initial_coords[:, :atom_coords.shape[1], :]
                
                expanded_mask = fixed_mask.expand(atom_coords.shape)
                
                # Reference coordinates are now pre-centered and will be transformed consistently 
                # with atom_coords during each diffusion step using the same transformations
                
                # Add fixed chains data to feats for use in forward pass
                feats["fixed_chains_ref_coords"] = initial_coords
                feats["fixed_chains_mask"] = expanded_mask
                
                # Keep minimal data for debugging
                fixed_chains_data = {
                    'total_steps': len(sigmas_and_gammas)
                }
                
                # DEBUG: Check initial geometry of fixed chains
                if initial_coords.shape[1] >= 2:
                    sample_bond = torch.norm(initial_coords[0, 1] - initial_coords[0, 0]).item()
                    # DEBUG: Initial sample bond distance: {sample_bond:.3f}Å")
                    
                    # Check if these look like reasonable protein coordinates
                    coord_range = initial_coords.max() - initial_coords.min()
                    # DEBUG: Coordinate range: {coord_range:.1f}Å")
                    
                    # Sample a few more consecutive bonds
                    if initial_coords.shape[1] >= 10:
                        sample_bonds = []
                        for i in range(min(5, initial_coords.shape[1]-1)):
                            bond = torch.norm(initial_coords[0, i+1] - initial_coords[0, i]).item()
                            sample_bonds.append(bond)
                        # DEBUG: First 5 consecutive bonds: {[f'{d:.3f}' for d in sample_bonds]}")
                
                
            except Exception as e:
                fixed_chains_data = None

        # gradually denoise
        for step_idx, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            random_R, random_tr = compute_random_augmentation(
                multiplicity, device=atom_coords.device, dtype=atom_coords.dtype
            )
            
            # Store un-rotated coordinates for trajectory (before random augmentation)
            if save_trajectory:
                atom_coords_centered = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
                trajectory_coords.append(atom_coords_centered[0].detach().cpu().clone())
            
            # GEOMETRY-PRESERVING FIXED CHAINS: Apply only translation, no rotation
            if fixed_chains_data is not None and "fixed_chains_ref_coords" in feats:
                # DEBUG: Step {step_idx}: Applying geometry-preserving transformation")
                reference_coords = feats["fixed_chains_ref_coords"]  
                fixed_mask = feats["fixed_chains_mask"]
                
                # DEBUG: Check geometry before transformation
                if step_idx == 0:
                    sample_fixed_bond = torch.norm(reference_coords[0, 1] - reference_coords[0, 0]).item()
                    # DEBUG: Step {step_idx}: Reference geometry sample bond: {sample_fixed_bond:.3f}Å")
                
                # For fixed atoms: apply ONLY translation to preserve internal geometry
                fixed_atoms = reference_coords
                
                # Transform non-fixed atoms normally (center + rotate + translate)
                non_fixed_mask = ~fixed_mask
                atom_coords_new = atom_coords.clone()
                
                if non_fixed_mask.any():
                    # Extract and transform non-fixed atoms
                    non_fixed_coords = torch.where(non_fixed_mask, atom_coords, torch.zeros_like(atom_coords))
                    non_fixed_mean = non_fixed_coords.sum(dim=-2, keepdims=True) / non_fixed_mask.sum(dim=-2, keepdims=True).float()
                    non_fixed_centered = (non_fixed_coords - non_fixed_mean) * non_fixed_mask.float()
                    non_fixed_transformed = torch.einsum("bmd,bds->bms", non_fixed_centered, random_R) + random_tr * non_fixed_mask.float()
                    
                    # For fixed atoms: apply translation to align with transformed structure
                    fixed_centroid = fixed_atoms.mean(dim=-2, keepdims=True)
                    non_fixed_centroid = non_fixed_transformed.sum(dim=-2, keepdims=True) / non_fixed_mask.sum(dim=-2, keepdims=True).float()
                    
                    # Translation to align fixed chain with transformed structure  
                    translation_offset = non_fixed_centroid - fixed_centroid + random_tr
                    fixed_atoms_aligned = fixed_atoms + translation_offset
                    
                    # Combine results: use transformed non-fixed and aligned fixed
                    atom_coords = torch.where(fixed_mask, fixed_atoms_aligned, non_fixed_transformed)
                    
                    # Update reference coordinates for next step
                    feats["fixed_chains_ref_coords"] = fixed_atoms_aligned
                else:
                    # If all atoms are fixed, apply only translation
                    atom_coords += random_tr
                    feats["fixed_chains_ref_coords"] = atom_coords
            else:
                # Standard processing when no fixed chains
                atom_coords_mean = atom_coords.mean(dim=-2, keepdims=True)
                atom_coords = atom_coords - atom_coords_mean
                atom_coords = (
                    torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
                )
            
            if atom_coords_denoised is not None:
                # Apply same transformation logic as for atom_coords
                if fixed_chains_data is not None and "fixed_chains_ref_coords" in feats:
                    # For fixed chains: keep them at reference positions
                    fixed_mask = feats["fixed_chains_mask"]
                    reference_coords = feats["fixed_chains_ref_coords"]
                    atom_coords_denoised = torch.where(fixed_mask, reference_coords, atom_coords_denoised)
                else:
                    # Standard transformation for denoised coordinates
                    atom_coords_denoised_mean = atom_coords_denoised.mean(dim=-2, keepdims=True)
                    atom_coords_denoised -= atom_coords_denoised_mean
                    atom_coords_denoised = (
                        torch.einsum("bmd,bds->bms", atom_coords_denoised, random_R) + random_tr
                    )
            if (
                steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
            ) and scaled_guidance_update is not None:
                scaled_guidance_update = torch.einsum(
                    "bmd,bds->bms", scaled_guidance_update, random_R
                )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            t_hat = sigma_tm * (1 + gamma)
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            noise_var = self.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = sqrt(noise_var) * torch.randn(shape, device=self.device)
            atom_coords_noisy = atom_coords + eps

            with torch.no_grad():
                atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
                sample_ids = torch.arange(multiplicity).to(atom_coords_noisy.device)
                sample_ids_chunks = sample_ids.chunk(
                    multiplicity % max_parallel_samples + 1
                )

                for sample_ids_chunk in sample_ids_chunks:
                    atom_coords_denoised_chunk = self.preconditioned_network_forward(
                        atom_coords_noisy[sample_ids_chunk],
                        t_hat,
                        network_condition_kwargs=dict(
                            multiplicity=sample_ids_chunk.numel(),
                            **network_condition_kwargs,
                        ),
                    )
                    atom_coords_denoised[sample_ids_chunk] = atom_coords_denoised_chunk

                # Skip fixed chains masking here - we'll apply it at the very end
                # This avoids issues with coordinate transformations corrupting the reference structure

                # Store un-rotated denoised coordinates for trajectory
                if save_trajectory and atom_coords_denoised is not None:
                    # Apply inverse transformation to remove random augmentation
                    inverse_R = random_R.transpose(-1, -2)  # Transpose for inverse rotation
                    atom_coords_denoised_unrotated = torch.einsum("bmd,bsd->bms", 
                                                                 atom_coords_denoised - random_tr, inverse_R)
                    # Center the un-rotated coordinates  
                    atom_coords_denoised_centered = atom_coords_denoised_unrotated - atom_coords_denoised_unrotated.mean(dim=-2, keepdims=True)
                    trajectory_denoised.append(atom_coords_denoised_centered[0].detach().cpu().clone())

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    # Compute energy of x_0 prediction
                    energy = torch.zeros(multiplicity, device=self.device)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if parameters["resampling_weight"] > 0:
                            component_energy = potential.compute(
                                atom_coords_denoised,
                                network_condition_kwargs["feats"],
                                parameters,
                            )
                            energy += parameters["resampling_weight"] * component_energy
                    energy_traj = torch.cat((energy_traj, energy.unsqueeze(1)), dim=1)

                    # Compute log G values
                    if step_idx == 0:
                        log_G = -1 * energy
                    else:
                        log_G = energy_traj[:, -2] - energy_traj[:, -1]

                    # Compute ll difference between guided and unguided transition distribution
                    if (
                        steering_args["physical_guidance_update"]
                        or steering_args["contact_guidance_update"]
                    ) and noise_var > 0:
                        ll_difference = (
                            eps**2 - (eps + scaled_guidance_update) ** 2
                        ).sum(dim=(-1, -2)) / (2 * noise_var)
                    else:
                        ll_difference = torch.zeros_like(energy)

                    # Compute resampling weights
                    resample_weights = F.softmax(
                        (ll_difference + steering_args["fk_lambda"] * log_G).reshape(
                            -1, steering_args["num_particles"]
                        ),
                        dim=1,
                    )

                # Compute guidance update to x_0 prediction
                if (
                    steering_args["physical_guidance_update"]
                    or steering_args["contact_guidance_update"]
                ) and step_idx < num_sampling_steps - 1:
                    guidance_update = torch.zeros_like(atom_coords_denoised)
                    for guidance_step in range(steering_args["num_gd_steps"]):
                        energy_gradient = torch.zeros_like(atom_coords_denoised)
                        for potential in potentials:
                            parameters = potential.compute_parameters(steering_t)
                            if (
                                parameters["guidance_weight"] > 0
                                and (guidance_step) % parameters["guidance_interval"]
                                == 0
                            ):
                                energy_gradient += parameters[
                                    "guidance_weight"
                                ] * potential.compute_gradient(
                                    atom_coords_denoised + guidance_update,
                                    network_condition_kwargs["feats"],
                                    parameters,
                                )
                        guidance_update -= energy_gradient
                    atom_coords_denoised += guidance_update
                    scaled_guidance_update = (
                        guidance_update
                        * -1
                        * self.step_scale
                        * (sigma_t - t_hat)
                        / t_hat
                    )

                if steering_args["fk_steering"] and (
                    (
                        step_idx % steering_args["fk_resampling_interval"] == 0
                        and noise_var > 0
                    )
                    or step_idx == num_sampling_steps - 1
                ):
                    resample_indices = (
                        torch.multinomial(
                            resample_weights,
                            resample_weights.shape[1]
                            if step_idx < num_sampling_steps - 1
                            else 1,
                            replacement=True,
                        )
                        + resample_weights.shape[1]
                        * torch.arange(
                            resample_weights.shape[0], device=resample_weights.device
                        ).unsqueeze(-1)
                    ).flatten()

                    atom_coords = atom_coords[resample_indices]
                    atom_coords_noisy = atom_coords_noisy[resample_indices]
                    atom_mask = atom_mask[resample_indices]
                    if atom_coords_denoised is not None:
                        atom_coords_denoised = atom_coords_denoised[resample_indices]
                    energy_traj = energy_traj[resample_indices]
                    if (
                        steering_args["physical_guidance_update"]
                        or steering_args["contact_guidance_update"]
                    ):
                        scaled_guidance_update = scaled_guidance_update[
                            resample_indices
                        ]
                    if token_repr is not None:
                        token_repr = token_repr[resample_indices]

            if self.alignment_reverse_diff:
                with torch.autocast("cuda", enabled=False):
                    atom_coords_noisy = weighted_rigid_align(
                        atom_coords_noisy.float(),
                        atom_coords_denoised.float(),
                        atom_mask.float(),
                        atom_mask.float(),
                    )

                atom_coords_noisy = atom_coords_noisy.to(atom_coords_denoised)

            # Fixed chains constraint is now applied at the beginning of each step
            # No need to apply here as it would interfere with the denoising process

            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
            atom_coords_next = (
                atom_coords_noisy + step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            # Fixed chains constraint will be applied at the beginning of the next step
            # No need to apply here to avoid double-application

            atom_coords = atom_coords_next

        # FINAL FIXED CHAINS APPLICATION - ensure fixed chains are preserved in final output
        if fixed_chains_data is not None and "fixed_chains_ref_coords" in feats:
            # DEBUG: Applying final fixed chains constraint")
            reference_coords = feats["fixed_chains_ref_coords"]
            fixed_mask = feats["fixed_chains_mask"]
            
            # DEBUG: Check geometry before final application
            if reference_coords.shape[1] >= 2:
                final_ref_bond = torch.norm(reference_coords[0, 1] - reference_coords[0, 0]).item()
                # DEBUG: Final reference bond: {final_ref_bond:.3f}Å")
            
            # Apply final fixed chains constraint to ensure they're preserved in output
            atom_coords = torch.where(
                fixed_mask,
                reference_coords,
                atom_coords
            )
            
            # DEBUG: Check geometry after final application
            if atom_coords.shape[1] >= 2:
                final_output_bond = torch.norm(atom_coords[0, 1] - atom_coords[0, 0]).item()
                # DEBUG: Final output bond: {final_output_bond:.3f}Å")

        # Prepare return dictionary
        result_dict = dict(sample_atom_coords=atom_coords, diff_token_repr=token_repr)
        
        # Add trajectory data if trajectory saving was enabled
        if save_trajectory:
            if trajectory_coords:
                result_dict["trajectory_coords"] = trajectory_coords
            if trajectory_denoised and not trajectory_decoded_only:
                result_dict["trajectory_denoised"] = trajectory_denoised
            elif trajectory_denoised and trajectory_decoded_only:
                # Only include denoised trajectory when decoded_only is requested
                result_dict["trajectory_denoised"] = trajectory_denoised
                
        return result_dict

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        feats,
        diffusion_conditioning,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0] // multiplicity

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        atom_coords = feats["coords"]

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        # Fixed chains logic is handled in the sampling loop for inference

        denoised_atom_coords = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            network_condition_kwargs={
                "s_inputs": s_inputs,
                "s_trunk": s_trunk,
                "feats": feats,
                "multiplicity": multiplicity,
                "diffusion_conditioning": diffusion_conditioning,
            },
        )

        return {
            "denoised_atom_coords": denoised_atom_coords,
            "sigmas": sigmas,
            "aligned_true_atom_coords": atom_coords,
        }

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
        filter_by_plddt=0.0,
    ):
        with torch.autocast("cuda", enabled=False):
            denoised_atom_coords = out_dict["denoised_atom_coords"].float()
            sigmas = out_dict["sigmas"].float()

            resolved_atom_mask_uni = feats["atom_resolved_mask"].float()

            if filter_by_plddt > 0:
                plddt_mask = feats["plddt"] > filter_by_plddt
                resolved_atom_mask_uni = resolved_atom_mask_uni * plddt_mask.float()

            resolved_atom_mask = resolved_atom_mask_uni.repeat_interleave(
                multiplicity, 0
            )

            align_weights = denoised_atom_coords.new_ones(denoised_atom_coords.shape[:2])
            atom_type = (
                torch.bmm(
                    feats["atom_to_token"].float(),
                    feats["mol_type"].unsqueeze(-1).float(),
                )
                .squeeze(-1)
                .long()
            )
            atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

            align_weights = (
                align_weights
                * (
                    1
                    + nucleotide_loss_weight
                    * (
                        torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                        + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
                    )
                    + ligand_loss_weight
                    * torch.eq(
                        atom_type_mult, const.chain_type_ids["NONPOLYMER"]
                    ).float()
                ).float()
            )

            atom_coords = out_dict["aligned_true_atom_coords"].float()
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords.detach(),
                denoised_atom_coords.detach(),
                align_weights.detach(),
                mask=feats["atom_resolved_mask"]
                .float()
                .repeat_interleave(multiplicity, 0)
                .detach(),
            )

            # Cast back
            atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
                denoised_atom_coords
            )

            # weighted MSE loss of denoised atom positions
            mse_loss = (
                (denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2
            ).sum(dim=-1)
            mse_loss = torch.sum(
                mse_loss * align_weights * resolved_atom_mask, dim=-1
            ) / (torch.sum(3 * align_weights * resolved_atom_mask, dim=-1) + 1e-5)

            # weight by sigma factor
            loss_weights = self.loss_weight(sigmas)
            mse_loss = (mse_loss * loss_weights).mean()

            total_loss = mse_loss

            # proposed auxiliary smooth lddt loss
            lddt_loss = self.zero
            if add_smooth_lddt_loss:
                lddt_loss = smooth_lddt_loss(
                    denoised_atom_coords,
                    feats["coords"],
                    torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                    + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                    coords_mask=resolved_atom_mask_uni,
                    multiplicity=multiplicity,
                )

                total_loss = total_loss + lddt_loss

            loss_breakdown = {
                "mse_loss": mse_loss,
                "smooth_lddt_loss": lddt_loss,
            }

        return {"loss": total_loss, "loss_breakdown": loss_breakdown}
