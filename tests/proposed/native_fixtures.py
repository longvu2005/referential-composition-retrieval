"""Deterministic native tensors for tests that exercise RCR without FAFA weights."""

import torch


class NativeFixture:
    metadata = {"version": "fixture-native", "cache_id": "fixture-native"}

    @property
    def native(self):
        return self

    def validate_samples(self, samples):
        pass

    def load_global_features(self, indices):
        return self.global_features[indices]

    def load_queries(self, samples, max_people):
        subjects = max(len(row["subjects"]) for row in samples)
        result = torch.zeros(len(samples), subjects, max_people, 5)
        for n, row in enumerate(samples):
            seed = sum(row["final_change"].encode())
            generator = torch.Generator().manual_seed(seed)
            result[n] = torch.randn(subjects, max_people, 5, generator=generator)
        return result

    def load_targets(self, indices, max_people):
        generator = torch.Generator().manual_seed(91)
        bank = torch.randn(100, max_people, 7, 5, generator=generator)
        return bank[indices]

    native_queries = load_queries
    native_targets = load_targets
