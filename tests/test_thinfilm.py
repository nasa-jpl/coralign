"""Unit tests for thinfilm.py."""
import unittest
import numpy as np
from math import isclose

from coralign.maskgen.util.thinfilm import solver, calc_complex_occulter


class TestThinfilm(unittest.TestCase):
    """
    Unit test suite for calc_complex_occulter().

    Compares against values from the commercial package the Essential Macleod.
    """
    
    def test_only_pmgi(self):
        """Test transmission of PMGI on FS."""
        lam = 600e-9
        d0 = 4*lam
        aoi = 0
        t_Ti_vec = [0]
        t_Ni_vec = [0]
        t_PMGI_vec = [100e-9]
        pol = 2
        [tCoef, rCoef] = calc_complex_occulter(lam, aoi, t_Ti_vec, t_Ni_vec,
                                          t_PMGI_vec, d0, pol)
        T = np.abs(tCoef)**2
        valueFromEssentialMacleod = 0.9431006949
        self.assertTrue(isclose(T, valueFromEssentialMacleod, rel_tol=1e-4))
        
    def test_ni_and_pmgi_s_pol(self):
        """Test transmission of PMGI on 95nm of Ni on FS, s polarization."""
        lam = 400e-9
        d0 = 4*lam
        aoi = 10
        t_Ti_vec = [0]
        t_Ni_vec = [95e-9]
        t_PMGI_vec = [0]
        pol = 0
        [tCoef, rCoef] = calc_complex_occulter(lam, aoi, t_Ti_vec, t_Ni_vec,
                                          t_PMGI_vec, d0, pol)
        T = np.abs(tCoef)**2
        # Value from Bala: 0.00087848574  Value from FALCO: 0.000878466587
        valueFromEssentialMacleod = 0.00087848574
        self.assertTrue(isclose(T, valueFromEssentialMacleod, rel_tol=1e-4))

    def test_ni_and_pmgi_p_pol(self):
        """Test transmission of PMGI on 95nm of Ni on FS, p polarization."""
        lam = 450e-9
        d0 = 4*lam
        aoi = 10
        t_Ti_vec = [0]
        t_Ni_vec = [95e-9]
        t_PMGI_vec = [30e-9]
        pol = 1
        [tCoef, rCoef] = calc_complex_occulter(lam, aoi, t_Ti_vec, t_Ni_vec,
                                          t_PMGI_vec, d0, pol)
        T = np.abs(tCoef)**2
        # Value from Bala: 0.00118382732,   Value from FALCO: 0.00118379
        valueFromEssentialMacleod = 0.00118382732
        self.assertTrue(isclose(T, valueFromEssentialMacleod, rel_tol=1e-4))

    def test_ni_and_pmgi_p_pol_at_another_wavelength(self):
        """Test transmission of PMGI on Ni on FS, p pol, at new wavelength."""
        lam = 550e-9
        d0 = 4*lam
        aoi = 10
        t_Ti_vec = [0]
        t_Ni_vec = [95e-9]
        t_PMGI_vec = [600e-9]
        pol = 1
        [tCoef, rCoef] = calc_complex_occulter(lam, aoi, t_Ti_vec, t_Ni_vec,
                                          t_PMGI_vec, d0, pol)
        T = np.abs(tCoef)**2
        # Value from Bala: 0.00121675706  Value from FALCO: 0.001216750339
        valueFromEssentialMacleod = 0.00121675706
        self.assertTrue(isclose(T, valueFromEssentialMacleod, rel_tol=1e-4))


class TestCalcComplexOcculterInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""
    
    def setUp(self):
        """Initialize good variable values for use in unit tests."""
        self.lam = 575e-9  # wavelength [meters]
        self.aoi = 5.5  # AOI [degrees]
        self.t_Ti_vec = [5e-9, 3e-9]
        self.t_Ni_vec = [80e-9, 100e-9]
        self.t_PMGI_vec = [500e-9, 650e-9]
        self.d0 = 2e-6
        self.pol = 2
        self.flagOPD = False
        self.sub = 'FS'
        
        # Make sure function runs smoothly first
        calc_complex_occulter(self.lam, self.aoi, self.t_Ti_vec, self.t_Ni_vec,
                              self.t_PMGI_vec, self.d0, self.pol,
                              flagOPD=self.flagOPD, SUBSTRATE=self.sub)
            
    def test_calc_complex_occulter_input_0(self):
        """Test incorrect inputs of calc_complex_occulter."""
        for lamBad in (1j, -0.1, 0, np.ones(2), 'string'):
            with self.assertRaises(ValueError):
                calc_complex_occulter(lamBad, self.aoi, self.t_Ti_vec,
                                      self.t_Ni_vec, self.t_PMGI_vec,
                                      self.d0, self.pol,
                                      flagOPD=self.flagOPD,
                                      SUBSTRATE=self.sub)
    
    def test_calc_complex_occulter_input_1(self):
        """Test incorrect inputs of calc_complex_occulter."""
        for aoiBad in (1j, -0.1, 90, 91, np.ones(2), 'string'):
            with self.assertRaises(ValueError):
                calc_complex_occulter(self.lam, aoiBad, self.t_Ti_vec,
                                      self.t_Ni_vec, self.t_PMGI_vec,
                                      self.d0, self.pol,
                                      flagOPD=self.flagOPD,
                                      SUBSTRATE=self.sub)

    def test_calc_complex_occulter_input_2(self):
        """Test incorrect inputs of calc_complex_occulter."""
        for t_Ti_vec_bad in (1j, -0.1, 90, 91, np.ones((2, 2)), 'string'):
            with self.assertRaises(ValueError):
                calc_complex_occulter(self.lam, self.aoi, t_Ti_vec_bad,
                                      self.t_Ni_vec, self.t_PMGI_vec,
                                      self.d0, self.pol,
                                      flagOPD=self.flagOPD,
                                      SUBSTRATE=self.sub)

    def test_calc_complex_occulter_input_3(self):
        """Test incorrect inputs of calc_complex_occulter."""
        for t_Ni_vec_bad in (1j, -0.1, 90, 91, np.ones((2, 2)), 'string'):
            with self.assertRaises(ValueError):
                calc_complex_occulter(self.lam, self.aoi, self.t_Ti_vec,
                                      t_Ni_vec_bad, self.t_PMGI_vec,
                                      self.d0, self.pol,
                                      flagOPD=self.flagOPD,
                                      SUBSTRATE=self.sub)
                
    def test_calc_complex_occulter_input_4(self):
        """Test incorrect inputs of calc_complex_occulter."""
        for t_PMGI_vec_bad in (1j, -0.1, 90, 91, np.ones((2, 2)), 'string'):
            with self.assertRaises(ValueError):
                calc_complex_occulter(self.lam, self.aoi, self.t_Ti_vec,
                                      self.t_Ni_vec, t_PMGI_vec_bad,
                                      self.d0, self.pol,
                                      flagOPD=self.flagOPD,
                                      SUBSTRATE=self.sub)

    def test_calc_complex_occulter_input_5(self):
        """Test incorrect inputs of calc_complex_occulter."""
        for d0Bad in (1j, -0.1, 0, np.ones(2), 'string'):
            with self.assertRaises(ValueError):
                calc_complex_occulter(self.lam, self.aoi, self.t_Ti_vec,
                                      self.t_Ni_vec, self.t_PMGI_vec,
                                      d0Bad, self.pol,
                                      flagOPD=self.flagOPD,
                                      SUBSTRATE=self.sub)

    def test_calc_complex_occulter_input_6(self):
        """Test incorrect inputs of calc_complex_occulter."""
        for polBad in (1j, -0.1, -1, 3, np.ones(2), 'string'):
            with self.assertRaises(ValueError):
                calc_complex_occulter(self.lam, self.aoi, self.t_Ti_vec,
                                      self.t_Ni_vec, self.t_PMGI_vec,
                                      self.d0, polBad,
                                      flagOPD=self.flagOPD,
                                      SUBSTRATE=self.sub)
                
    def test_calc_complex_occulter_input_7(self):
        """Test incorrect inputs of calc_complex_occulter."""
        for flagOPDBad in (1j, -0.1, 0, 1, np.ones(2), 'string'):
            with self.assertRaises(TypeError):
                calc_complex_occulter(self.lam, self.aoi, self.t_Ti_vec,
                                      self.t_Ni_vec, self.t_PMGI_vec,
                                      self.d0, self.pol,
                                      flagOPD=flagOPDBad,
                                      SUBSTRATE=self.sub)
 
    def test_calc_complex_occulter_input_8(self):
        """Test incorrect inputs of calc_complex_occulter."""
        for subBad in (1j, -0.1, 90, 91, np.ones(2)):
            with self.assertRaises(TypeError):
                calc_complex_occulter(self.lam, self.aoi, self.t_Ti_vec,
                                      self.t_Ni_vec, self.t_PMGI_vec,
                                      self.d0, self.pol,
                                      flagOPD=self.flagOPD,
                                      SUBSTRATE=subBad)
                
    def test_calc_complex_occulter_input_8b(self):
        """Test incorrect inputs of calc_complex_occulter."""
        for subBad in ('wrongString'):
            with self.assertRaises(ValueError):
                calc_complex_occulter(self.lam, self.aoi, self.t_Ti_vec,
                                      self.t_Ni_vec, self.t_PMGI_vec,
                                      self.d0, self.pol,
                                      flagOPD=self.flagOPD,
                                      SUBSTRATE=subBad)


class TestSolverInputFailure(unittest.TestCase):
    """Test suite for valid function inputs."""
    
    def setUp(self):
        """Initialize good variable values for use in unit tests."""
        self.n = [1.5, 1.3, 1.2, 1.0]
        self.d0 = [100e-9, 250e-9]
        self.theta = 5.5 * (np.pi/180.)  # AOI [radians]
        self.lam = 575e-9  # wavelength [meters]
        self.tetm = True
        
        # Make sure function runs smoothly first
        solver(self.n, self.d0, self.theta, self.lam, self.tetm)
    
    def test_solver_mismatching_n_d0(self):
        """Test incorrect inputs of solver."""
        with self.assertRaises(ValueError):
            solver(self.n, [95e-9], self.theta, self.lam, self.tetm)

    def test_solver_input_0(self):
        """Test incorrect inputs of solver."""
        for nBad in (1j, np.ones((5, 1)), np.ones((5, 2, 3)), 'string'):
            with self.assertRaises(ValueError):
                solver(nBad, self.d0, self.theta, self.lam, self.tetm)
    
    def test_solver_input_1(self):
        """Test incorrect inputs of solver."""
        for d0Bad in (1j, np.ones((5, 1)), np.ones((5, 2, 3)), 'string'):
            with self.assertRaises(ValueError):
                solver(self.n, d0Bad, self.theta, self.lam, self.tetm)
    
    def test_solver_input_2(self):
        """Test incorrect inputs of solver."""
        for thetaBad in (1j, -0.1, np.pi/2, np.ones(2), 'string'):
            with self.assertRaises(ValueError):
                solver(self.n, self.d0, thetaBad, self.lam, self.tetm)
                
    def test_solver_input_3(self):
        """Test incorrect inputs of solver."""
        for lamBad in (1j, -0.1, 0, np.ones(2), 'string'):
            with self.assertRaises(ValueError):
                solver(self.n, self.d0, self.theta, lamBad, self.tetm)
    
    def test_solver_input_4(self):
        """Test incorrect inputs of solver."""
        for tetmBad in (1j, -0.1, 0, 1, np.ones(2), 'string'):
            with self.assertRaises(TypeError):
                solver(self.n, self.d0, self.theta, self.lam, tetmBad)


if __name__ == '__main__':
    unittest.main()
