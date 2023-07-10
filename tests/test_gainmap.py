"""Unit test suite for gainmap module."""
import unittest
import os
import numpy as np

import coralign.gainmap.gainmap as gm


class TestGainmap(unittest.TestCase):
    """Unit tests for gainmap.py."""

    def setUp(self):
        """Set up variables for communal use."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.fnRefCubeLinear = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeLinear.fits')
        self.fnRefCube = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeQuadratic.fits')
        self.fnRefCommandVec = os.path.join(self.localpath, 'testdata',
                                            'ut_refCommandVec.fits')
        self.nActs = 48
        self.a = 1/100  # quadratic coefficient for ut_refCubeQuadratic.fits
        self.b = 3   # linear coefficient for ut_refCubeQuadratic.fits
        self.c = 0  # DC coefficient for ut_refCubeQuadratic.fits

    def test_compute_gainmap(self):
        """Test the accuracy of compute_gainmap()."""
        nActs = self.nActs
        commandMap = np.linspace(0, 100, nActs**2).reshape((nActs, nActs))
        gainMapExpected = 2*self.a*commandMap + self.b
        gainMap = gm.compute_gainmap(commandMap,
                                     self.fnRefCube,
                                     self.fnRefCommandVec,
                                     )
        self.assertTrue(np.allclose(gainMap,
                                    gainMapExpected,
                                    atol=np.finfo(float).eps,
                                    )
                        )

    def test_compute_starting_commands_for_flight(self):
        """Test the accuracy of compute_starting_commands_for_flight()."""
        flatCommandMap = 50 * np.ones((self.nActs, self.nActs))
        deltaHeightMap = 100 * np.ones((self.nActs, self.nActs))

        heightMap = (self.a*flatCommandMap**2 + self.b*flatCommandMap +
                     deltaHeightMap)
        startingCommandMapTrue = ((-self.b + np.sqrt(self.b*self.b +
                                                     4*self.a*heightMap)) /
                                  (2*self.a))

        startingCommandMap = gm.compute_starting_commands_for_flight(
            flatCommandMap,
            deltaHeightMap,
            self.fnRefCube,
            self.fnRefCommandVec)

        self.assertTrue(np.allclose(startingCommandMapTrue,
                                    startingCommandMap,
                                    atol=1e-1,
                                    )
                        )

    def test_compute_starting_commands_for_flight_lower_bound(self):
        """Test the lower bound of compute_starting_commands_for_flight()."""
        flatCommandMap = 50 * np.ones((self.nActs, self.nActs))
        deltaHeightMap = -1e3 * np.ones((self.nActs, self.nActs))

        startingCommandMapTrue = np.zeros((self.nActs, self.nActs))

        startingCommandMap = gm.compute_starting_commands_for_flight(
            flatCommandMap,
            deltaHeightMap,
            self.fnRefCube,
            self.fnRefCommandVec)

        self.assertTrue(np.allclose(startingCommandMapTrue,
                                    startingCommandMap,
                                    atol=1e-4,
                                    )
                        )

    def test_compute_starting_commands_for_flight_upper_bound(self):
        """Test the lower bound of compute_starting_commands_for_flight()."""
        flatCommandMap = 50 * np.ones((self.nActs, self.nActs))
        deltaHeightMap = 1e3 * np.ones((self.nActs, self.nActs))

        startingCommandMapTrue = 100 * np.ones((self.nActs, self.nActs))

        startingCommandMap = gm.compute_starting_commands_for_flight(
            flatCommandMap,
            deltaHeightMap,
            self.fnRefCube,
            self.fnRefCommandVec)

        self.assertTrue(np.allclose(startingCommandMapTrue,
                                    startingCommandMap,
                                    atol=1e-4,
                                    )
                        )


class TestGainmapInputFailure(unittest.TestCase):
    """Unit tests for inputs failures of gainmap.py."""

    def setUp(self):
        """Set up variables for communal use."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.fnRefCubeLinear = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeQuadratic.fits')
        self.fnRefCommandVec = os.path.join(self.localpath, 'testdata',
                                            'ut_refCommandVec.fits')

    def test_compute_gainmap_input(self):
        """Test compute_gainmap() with bad inputs."""
        # filename errors are handled directly by fits.getdata
        for commandMapBad in (1j, np.ones((48, 10)), np.ones((48, 10)), 's'):
            with self.assertRaises(ValueError):
                gm.compute_gainmap(commandMapBad,
                                   self.fnRefCubeLinear,
                                   self.fnRefCommandVec,
                                   )

    def test_compute_starting_commands_for_flight_inputs(self):
        """Test compute_starting_commands_for_flight() with bad inputs."""
        flatCommandMap = np.ones((48, 48))
        deltaHeightMap = np.ones((48, 48))

        for flatCommandMapBad in (1j, np.ones((48, 10)), np.ones((48, 10))):
            with self.assertRaises(ValueError):
                gm.compute_starting_commands_for_flight(flatCommandMapBad,
                                                        deltaHeightMap,
                                                        self.fnRefCubeLinear,
                                                        self.fnRefCommandVec)

        for deltaHeightMapBad in (1j, np.ones((48, 10)), np.ones((48, 10))):
            with self.assertRaises(ValueError):
                gm.compute_starting_commands_for_flight(flatCommandMap,
                                                        deltaHeightMapBad,
                                                        self.fnRefCubeLinear,
                                                        self.fnRefCommandVec)


if __name__ == '__main__':
    unittest.main()
