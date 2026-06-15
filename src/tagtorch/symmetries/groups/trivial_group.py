from typing import List

############### Trivial group #######################

class TrivialGroup:
    """
    The trivial group consisting of the identity element.

    This class implements the trivial group {e} with only the identity element.
    All operations return the identity element.
    """
    order = 1
    abelian = True

    def enumerate_elements(self) -> List[int]:
        """Return all elements in the group.

        Returns
        -------
        List[int]
            List containing the single identity element.
        """
        return [1]

    def sample(self) -> int:
        """Sample a random element of the trivial group.

        Returns
        -------
        int
            Always returns 1, the only element in the trivial group.
        """
        return 1

    def compose(self, g1: int, g2: int) -> int:
        """Compose two group elements.

        Parameters
        ----------
        g1 : int
            First group element (always 1).
        g2 : int
            Second group element (always 1).

        Returns
        -------
        int
            The identity element (1).
        """
        return 1

    def inverse(self, g: int) -> int:
        """Compute the inverse of a group element.

        Parameters
        ----------
        g : int
            Group element (always 1).

        Returns
        -------
        int
            The inverse element (always 1).
        """
        return 1

    def identity(self) -> int:
        """Return the identity element.

        Returns
        -------
        int
            The identity element (1).
        """
        return 1
    


        
