from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    @abstractmethod
    def load_weights(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode_image(self, image_tensor):
        raise NotImplementedError

    @abstractmethod
    def extract_intermediate(self, image_tensor, layers):
        raise NotImplementedError

    @abstractmethod
    def get_feature_spec(self) -> dict:
        raise NotImplementedError
