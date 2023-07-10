"""Unit test suite for gainmap module."""
import unittest
import os
import numpy as np

import coralign.gainmap.gm_util as gmu


class TestGainmapUtils(unittest.TestCase):
    """Unit tests for gm_util.py."""

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

    def test_diff_of_vector_(self):
        """Test differencing part of compute_diff_and_mean_of_vector()."""
        diffVec, _ = gmu.compute_diff_and_mean_of_vector(self.fnRefCommandVec)
        expectedVec = 20*np.ones(5)

        self.assertTrue(np.allclose(diffVec,
                                    expectedVec,
                                    atol=np.finfo(float).eps,
                                    )
                        )

    def test_mean_of_vector_(self):
        """Test averaging part of compute_diff_and_mean_of_vector()."""
        _, meanVec = gmu.compute_diff_and_mean_of_vector(self.fnRefCommandVec)
        expectedVec = np.arange(10, 100, 20)

        self.assertTrue(np.allclose(meanVec,
                                    expectedVec,
                                    atol=np.finfo(float).eps,
                                    )
                        )

    def test_compute_gain_cube_from_height_cube_linear(self):
        """Test functionality of test_compute_gain_cube_from_height_cube()."""
        gainCube = gmu.compute_gain_cube_from_height_cube(
            self.fnRefCubeLinear,
            self.fnRefCommandVec
        )
        expectedVec = np.ones((5, self.nActs, self.nActs))

        self.assertTrue(np.allclose(gainCube,
                                    expectedVec,
                                    atol=np.finfo(float).eps,
                                    )
                        )

    def test_compute_gain_cube_from_height_cube_quadratic(self):
        """Test functionality of test_compute_gain_cube_from_height_cube()."""
        gainCube = gmu.compute_gain_cube_from_height_cube(
            self.fnRefCube,
            self.fnRefCommandVec
        )

        nSlices = 5
        commands = np.arange(10, 100, 20)
        gains = 2*self.a*commands + self.b
        expectedVec = np.ones((nSlices, self.nActs, self.nActs))
        for ii in range(nSlices):
            expectedVec[ii, :, :] = gains[ii] * expectedVec[ii, :, :]

        self.assertTrue(np.allclose(gainCube,
                                    expectedVec,
                                    atol=np.finfo(float).eps,
                                    )
                        )

    def test_compute_delta_height_map_from_command_maps(self):
        """Test accuracy of compute_delta_height_map_from_command_maps()."""
        commandBefore = 25.2
        commandAfter = 45.5
        commandMapBefore = commandBefore * np.ones((self.nActs, self.nActs))
        commandMapAfter = commandAfter * np.ones((self.nActs, self.nActs))
        deltaHeight = ((self.a*commandAfter**2 + self.b*commandAfter) -
                       (self.a*commandBefore**2 + self.b*commandBefore))
        deltaHeightMapTrue = deltaHeight * np.ones((self.nActs, self.nActs))

        deltaHeightMap = gmu.compute_delta_height_map_from_command_maps(
            commandMapBefore,
            commandMapAfter,
            self.fnRefCube,
            self.fnRefCommandVec)

        self.assertTrue(np.allclose(deltaHeightMapTrue,
                                    deltaHeightMap,
                                    atol=1e-4,
                                    )
                        )

    def test_compute_commands_for_height_map(self):
        """Test accuracy of compute_commands_for_height_map()."""
        nActs = self.nActs
        heightMap = np.linspace(0, 400, nActs**2).reshape((nActs, nActs))
        commandMapTrue = (-self.b + np.sqrt(self.b*self.b +
                                            4*self.a*heightMap)) / (2*self.a)

        commandMap = gmu.compute_commands_for_height_map(
            heightMap,
            self.fnRefCube,
            self.fnRefCommandVec,
        )

        self.assertTrue(np.allclose(commandMapTrue, commandMap, atol=1e-1))

    def test_compute_heights_for_command_map(self):
        """Test accuracy of compute_heights_for_command_map()."""
        nActs = self.nActs
        commandMap = np.linspace(0, 100, nActs**2).reshape((nActs, nActs))
        heightMapTrue = self.a*commandMap**2 + self.b*commandMap + self.c
        heightMap = gmu.compute_heights_for_command_map(
            commandMap,
            self.fnRefCube,
            self.fnRefCommandVec,
        )

        self.assertTrue(np.allclose(heightMapTrue, heightMap, atol=1e-4))


class TestGainmapInputFailure(unittest.TestCase):
    """Unit tests for inputs failures of gainmap.py."""

    def setUp(self):
        """Set up variables for communal use."""
        self.localpath = os.path.dirname(os.path.abspath(__file__))
        self.fnRefCube = os.path.join(self.localpath, 'testdata',
                                      'ut_refCubeQuadratic.fits')
        self.fnRefCommandVec = os.path.join(self.localpath, 'testdata',
                                            'ut_refCommandVec.fits')

    def test_verify_vector_has_unique_values(self):
        """Test that error is raised if input is not a unique-valued vector."""
        # Check that good input does not fail
        _ = gmu.verify_vector_has_unique_values((2, 3, 4))

        # Exception for non-1D array
        for badVec in (1j, 'string', np.ones((3, 3))):
            with self.assertRaises(ValueError):
                gmu.verify_vector_has_unique_values(badVec)

        # Exception for nonunique values
        for badVec in ((1, 1, 2)):
            with self.assertRaises(ValueError):
                gmu.verify_vector_has_unique_values(badVec)

    def test_verify_vector_is_presorted(self):
        """Test that error is raised if input not a sorted vector."""
        # Check that good input does not fail
        _ = gmu.verify_vector_is_presorted((2, 3, 4))

        # Exception for non-1D array
        for badVec in (1j, 'string', np.ones((3, 3))):
            with self.assertRaises(ValueError):
                gmu.verify_vector_is_presorted(badVec)

        # Exception for unsorted array
        for badVec in ((2, 1), np.arange(0, -10, -1)):
            with self.assertRaises(ValueError):
                gmu.verify_vector_is_presorted(badVec)

    def test_not_a_cube(self):
        """Test compute_gain_cube_from_height_cube() with a bad input."""
        with self.assertRaises(ValueError):
            gmu.compute_gain_cube_from_height_cube(self.fnRefCommandVec,
                                                   self.fnRefCommandVec
                                                   )

    def test_mismatched_vector_length_and_cube_height(self):
        """Test for exception if vector and datacube are incompatible."""
        fnLongVec = os.path.join(self.localpath, 'testdata',
                                'ut_vector_with_too_many_values.fits')
        with self.assertRaises(ValueError):
            gmu.compute_gain_cube_from_height_cube(
                self.fnRefCube,
                fnLongVec,
            )

    def test_compute_delta_height_map_from_command_maps_inputs(self):
        """Test bad inputs for compute_delta_height_map_from_command_maps()."""
        commandMapBefore = np.ones((48, 48))
        commandMapAfter = 2*np.ones((48, 48))

        # Check that no issues are raised with good inputs
        gmu.compute_delta_height_map_from_command_maps(
            commandMapBefore,
            commandMapAfter,
            self.fnRefCube,
            self.fnRefCommandVec
        )

        for commandMapBeforeBad in (1j, np.ones((48, 10)), np.ones((48, 10)),
                                    'string'):
            with self.assertRaises(ValueError):
                gmu.compute_delta_height_map_from_command_maps(
                    commandMapBeforeBad,
                    commandMapAfter,
                    self.fnRefCube,
                    self.fnRefCommandVec
                )
        for commandMapAfterBad in (1j, np.ones((48, 10)), np.ones((48, 10)),
                                   'string'):
            with self.assertRaises(ValueError):
                gmu.compute_delta_height_map_from_command_maps(
                    commandMapBefore,
                    commandMapAfterBad,
                    self.fnRefCube,
                    self.fnRefCommandVec
                )

    def test_compute_commands_for_height_map_inputs(self):
        """Test compute_commands_for_height_map() with bad inputs."""
        # Check that no issues are raised with good inputs
        gmu.compute_commands_for_height_map(np.ones((48, 48)),
                                            self.fnRefCube,
                                            self.fnRefCommandVec)

        # Wrong type or shape for heightMap
        for heightMapBad in (1j, np.ones((48, 10)), np.ones((48, 10)), 's'):
            with self.assertRaises(ValueError):
                gmu.compute_commands_for_height_map(heightMapBad,
                                                    self.fnRefCube,
                                                    self.fnRefCommandVec)

        # Out of bounds
        for heightMapBad in (-1*np.ones((48, 48)), 1e6*np.ones((48, 48))):
            with self.assertRaises(ValueError):
                gmu.compute_commands_for_height_map(heightMapBad,
                                                    self.fnRefCube,
                                                    self.fnRefCommandVec)

    def test_compute_heights_for_command_map_inputs(self):
        """Test compute_heights_for_command_map_inputs() with bad inputs."""
        # Check that no issues are raised with good inputs
        gmu.compute_heights_for_command_map(np.ones((48, 48)),
                                            self.fnRefCube,
                                            self.fnRefCommandVec)

        # Wrong type or shape for commandMap
        for commandMapBad in (1j, np.ones((48, 10)), np.ones((48, 10)), 's'):
            with self.assertRaises(ValueError):
                gmu.compute_heights_for_command_map(commandMapBad,
                                                    self.fnRefCube,
                                                    self.fnRefCommandVec)

        # Out of bounds
        for commandMapBad in (-1*np.ones((48, 48)), 101*np.ones((48, 48))):
            with self.assertRaises(ValueError):
                gmu.compute_heights_for_command_map(commandMapBad,
                                                    self.fnRefCube,
                                                    self.fnRefCommandVec)


if __name__ == '__main__':
    unittest.main()
