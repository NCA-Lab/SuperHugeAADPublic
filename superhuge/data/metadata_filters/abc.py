from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..metadata_processing.data import (
    ClassifyMetadataElement,
    MetadataElementType,
    RegressionMetadataElement,
)


class MetadataFilter(ABC, Generic[MetadataElementType]):
    @abstractmethod
    def __call__(
        self,
        metadata_element: MetadataElementType | None,
    ) -> MetadataElementType | None:
        """
        __call__ call method of a metadata filter.

        :param metadata_element: metadata_element to be filtered. Should be an instance of MetadataElement.
        :type metadata_element: MetadataElement | None
        :return: fitlered metadata_element (can be modified according to the rule of the filter). Or `None`, if
        to discard this metadata_element.
        :rtype: MetadataElement | None
        """
        pass


class ClassifyMetadataFilter(MetadataFilter[ClassifyMetadataElement]):
    """
    Base class for classification metadata filters.
    Does nothing, just for identification.
    """

    pass


class RegressionMetadataFilter(MetadataFilter[RegressionMetadataElement]):
    """
    Base class for regression metadata filters.
    Does nothing, just for identification.
    """

    pass
