"""RGI input round-trips, pipeline preservation and JAX sampler wiring."""

import json
from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from alphafold3.common import folding_input
from alphafold3.data import pipeline
from alphafold3.model.network import diffusion_head


def fold_input():
    return folding_input.Input.from_json(
        json.dumps(
            {
                "name": "rgi",
                "dialect": "alphafold3",
                "version": 4,
                "modelSeeds": [0],
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "AG",
                            "unpairedMsa": "",
                            "pairedMsa": "",
                            "templates": [],
                            "conformer_restraints": True,
                        }
                    },
                    {
                        "protein": {
                            "id": "B",
                            "sequence": "AG",
                            "unpairedMsa": "",
                            "pairedMsa": "",
                            "templates": [],
                        }
                    },
                    {
                        "rna": {
                            "id": "R",
                            "sequence": "AC",
                            "unpairedMsa": "",
                            "conformer_restraints": True,
                        }
                    },
                    {
                        "dna": {
                            "id": "D",
                            "sequence": "AC",
                            "conformer_restraints": True,
                        }
                    },
                    {
                        "ligand": {
                            "id": "L",
                            "smiles": "F/C=C/Cl",
                            "conformer_restraints": True,
                        }
                    },
                ],
                "restraints_config": {
                    "verbose": True,
                    "distance_restraints_config": [
                        {
                            "atom_selection1": "chain A",
                            "atom_selection2": "chain B",
                            "harmonic": {"target_distance": 25.0},
                        }
                    ],
                },
            }
        )
    )


def test_roundtrip_keeps_config_and_distinct_chain_opt_ins():
    original = fold_input()
    restored = folding_input.Input.from_json(original.to_json())
    assert restored.restraints_config == original.restraints_config
    assert [c.conformer_restraints for c in restored.chains] == [
        True,
        False,
        True,
        True,
        True,
    ]
    assert len(restored.chains) == 5
    assert original.chains[0].fill_missing_fields().conformer_restraints
    assert original.chains[2].fill_missing_fields().conformer_restraints


def test_data_pipeline_preserves_polymer_opt_ins():
    pipe = object.__new__(pipeline.DataPipeline)
    original = fold_input()
    processed = pipe.process(original)
    assert [c.conformer_restraints for c in processed.chains] == [
        True,
        False,
        True,
        True,
        True,
    ]
    assert processed.restraints_config == original.restraints_config


@pytest.mark.parametrize("steering", [False, True])
@pytest.mark.parametrize("use_rgi", [False, True])
def test_sampler_preserves_finite_coordinates_and_rgi_order(
    monkeypatch, steering, use_rgi
):
    from intellifold.steering import guidance

    monkeypatch.setattr(guidance, "default_groups", lambda: [])
    monkeypatch.setattr(
        guidance, "apply_guidance", lambda positions, *a, **kw: positions + 2.0
    )
    batch = SimpleNamespace(
        predicted_structure_info=SimpleNamespace(atom_mask=jnp.ones((1, 2)))
    )
    config = diffusion_head.SampleConfig(
        steps=2, num_samples=2, step_scale=1.0, steering_enabled=steering
    )

    class Restraints:
        def minimize_gpu(self, positions, sigma, step):
            expected = 2.0 if steering else 0.0
            valid = jnp.all(positions == expected)
            # The two scan steps must receive the previous, pre-churn sigma.
            levels = diffusion_head.noise_schedule(jnp.linspace(0.0, 1.0, 3))
            valid &= jnp.isclose(sigma, levels[step])
            return jnp.full_like(positions, jnp.where(valid, 7.0 + step, -1000.0))

    def denoise(positions, sigma):
        return jnp.zeros_like(positions)

    @hk.transform
    def forward():
        return diffusion_head.sample(
            denoising_step=denoise,
            batch=batch,
            key=hk.next_rng_key(),
            config=config,
            steering={} if steering else None,
            restraints=Restraints() if use_rgi else None,
        )

    key = jax.random.PRNGKey(3)
    params = forward.init(key)
    result = jax.jit(forward.apply)(params, key)
    assert np.isfinite(result["atom_positions"]).all()
    # Last-step Euler with unit scale approaches the corrected denoised coordinates.
    expected = 8.0 if use_rgi else (2.0 if steering else 0.0)
    np.testing.assert_allclose(result["atom_positions"], expected, atol=0.02)
