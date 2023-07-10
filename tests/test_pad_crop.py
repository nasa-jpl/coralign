"""
Unit tests for pad_crop.py
"""
import numpy as np
import unittest

from coralign.util.pad_crop import pad_crop, offcenter_crop


class TestPadCrop(unittest.TestCase):
    """
    Unit test suite for pad_crop()

    This will have lots of special cases to handle odd/even sizing and
    truncating vs. non-truncating
    """

    # Success tests (non-truncating)
    def test_insert_all_size_odd(self):
        """Check insert behavior, (o,o) --> (o,o)"""
        out = pad_crop(np.ones((3, 3)), (5, 5))
        test = np.array([[0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_all_size_even(self):
        """Check insert behavior, (e,e) --> (e,e)"""
        out = pad_crop(np.ones((2, 2)), (4, 4))
        test = np.array([[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 0]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_first_index_odd_size_even(self):
        """Check insert behavior, (o,e) --> (e,e)"""
        out = pad_crop(np.ones((3, 2)), (4, 4))
        test = np.array([[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_second_index_odd_size_even(self):
        """Check insert behavior, (e,o) --> (e,e)"""
        out = pad_crop(np.ones((2, 3)), (4, 4))
        test = np.array([[0, 0, 0, 0],
                         [0, 1, 1, 1],
                         [0, 1, 1, 1],
                         [0, 0, 0, 0]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_first_index_even_size_odd(self):
        """Check insert behavior, (e,o) --> (o,o)"""
        out = pad_crop(np.ones((4, 3)), (5, 5))
        test = np.array([[0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_second_index_even_size_odd(self):
        """Check insert behavior, (o,e) --> (o,o)"""
        out = pad_crop(np.ones((3, 4)), (5, 5))
        test = np.array([[0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]])
        self.assertTrue((out == test).all())
        pass


    # Truncating
    def test_insert_all_size_odd_trunc(self):
        """Check insert behavior, (o,o) --> (o,o), truncating"""
        tmat = np.outer(np.arange(0, 10, 2), np.arange(0, 15, 3))
        out = pad_crop(tmat, (3, 3))
        test = np.array([[6, 12, 18],
                         [12, 24, 36],
                         [18, 36, 54]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_all_size_even_trunc(self):
        """Check insert behavior, (e,e) --> (e,e), truncating"""
        tmat = np.outer(np.arange(0, 8, 2), np.arange(0, 12, 3))
        out = pad_crop(tmat, (2, 2))
        test = np.array([[6, 12],
                         [12, 24]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_first_index_odd_size_even_trunc(self):
        """Check insert behavior, (e,e) --> (o,e), truncating"""
        tmat = np.outer(np.arange(0, 8, 2), np.arange(0, 12, 3))
        out = pad_crop(tmat, (3, 2))
        test = np.array([[6, 12],
                         [12, 24],
                         [18, 36]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_second_index_odd_size_even_trunc(self):
        """Check insert behavior, (e,e) --> (e,o), truncating"""
        tmat = np.outer(np.arange(0, 8, 2), np.arange(0, 12, 3))
        out = pad_crop(tmat, (2, 3))
        test = np.array([[6, 12, 18],
                         [12, 24, 36]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_first_index_even_size_odd_trunc(self):
        """Check insert behavior, (e,o) --> (o,o), truncating"""
        tmat = np.outer(np.arange(0, 10, 2), np.arange(0, 15, 3))
        out = pad_crop(tmat, (4, 3))
        test = np.array([[0, 0, 0],
                         [6, 12, 18],
                         [12, 24, 36],
                         [18, 36, 54]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_second_index_even_size_odd_trunc(self):
        """Check insert behavior, (o,e) --> (o,o), truncating"""
        tmat = np.outer(np.arange(0, 10, 2), np.arange(0, 15, 3))
        out = pad_crop(tmat, (3, 4))
        test = np.array([[0, 6, 12, 18],
                         [0, 12, 24, 36],
                         [0, 18, 36, 54]])
        self.assertTrue((out == test).all())
        pass

    # Mixed
    def test_insert_first_large_second_small(self):
        """Check insert behavior, truncating second axis only"""
        out = pad_crop(np.ones((4, 4)), (6, 2))
        test = np.array([[0, 0],
                         [1, 1],
                         [1, 1],
                         [1, 1],
                         [1, 1],
                         [0, 0]])
        self.assertTrue((out == test).all())
        pass

    def test_insert_first_small_second_large(self):
        """Check insert behavior, truncating first axis only"""
        out = pad_crop(np.ones((4, 4)), (2, 6))
        test = np.array([[0, 1, 1, 1, 1, 0],
                         [0, 1, 1, 1, 1, 0]])
        self.assertTrue((out == test).all())
        pass


    # Other success
    def test_transitivity(self):
        """
        Verify that successive pad_crops always end up at the same location.
        Should be no path dependence
        """
        for midsize in [(3, 3), (3, 4), (8, 7), (6, 6), (2, 2), (8, 8)]:
            out1a = pad_crop(np.ones((2, 2)), midsize)
            out1b = pad_crop(out1a, (8, 8))
            out2 = pad_crop(np.ones((2, 2)), (8, 8))
            self.assertTrue((out1b == out2).all())
            pass
        pass

    def test_dtype_passed(self):
        """Check pad_crop maintains data type"""
        for dt in [bool, np.int32, np.float32, np.float64,
                   np.complex64, np.complex128]:
            inm = np.ones((2, 2), dtype=dt)
            out = pad_crop(inm, (4, 4))
            self.assertTrue(out.dtype == inm.dtype)
            pass
        pass



    # Failure tests
    def test_arr0_2darray(self):
        """Check input array type valid"""
        for arr0 in [(2, 2), np.zeros((2,)), np.zeros((2, 2, 2))]:
            with self.assertRaises(TypeError):
                pad_crop(arr0, (4, 4))
                pass
            pass
        pass

    def test_outsize_2elem_list(self):
        """Check outsize is 2-element list"""
        for outsize in [(4,), (4, 4, 4), [], None, 4]:
            with self.assertRaises(TypeError):
                pad_crop(np.zeros((2, 2)), outsize)
                pass
            pass
        pass

    def test_outsize_has_non_positive_int(self):
        """Check outsize elements are positive integers"""
        for outsize in [(0, 4), (-5, 4), (6.3, 4), (4.0, 4)]:
            with self.assertRaises(TypeError):
                pad_crop(np.zeros((2, 2)), outsize)
                pass
            pass
        pass


class TestOffcenterCrop(unittest.TestCase):
    """Test suite for offcenter_crop()."""

    def setUp(self):
        """Define reused variables."""
        self.image = np.ones((15, 34))
        self.pixel_count_across = 20
        self.output_center_x = 4
        self.output_center_y = 3

    def test_offcenter_crop_input_0(self):
        """Test bad inputs to offcenter_crop."""
        for badVal in (-1, 0, 1, 1.5, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                offcenter_crop(badVal,
                               self.pixel_count_across,
                               self.output_center_x,
                               self.output_center_y)

    def test_offcenter_crop_input_1(self):
        """Test bad inputs to offcenter_crop."""
        for badVal in (-1, 0, 1.5, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                offcenter_crop(self.image,
                               badVal,
                               self.output_center_x,
                               self.output_center_y)

    def test_offcenter_crop_input_2(self):
        """Test bad inputs to offcenter_crop."""
        for badVal in (1j, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                offcenter_crop(self.image,
                               self.pixel_count_across,
                               badVal,
                               self.output_center_y)

    def test_offcenter_crop_input_3(self):
        """Test bad inputs to offcenter_crop."""
        for badVal in (1j, (1, 2), np.ones((3, 3, 3)), 'asdf'):
            with self.assertRaises(TypeError):
                offcenter_crop(self.image,
                               self.pixel_count_across,
                               self.output_center_x,
                               badVal)

    def test_odd_in_odd_out_stay_centered(self):
        """Test a particular usage case."""
        image = np.zeros((5, 5))
        image[2::, 2::] = 1
        pupil_center_x = 3
        pupil_center_y = 3
        pixel_count_across = 7
        cropped_image = offcenter_crop(image, pixel_count_across,
                                       pupil_center_y, pupil_center_x)
        answer = pad_crop(np.ones((3, 3)), (7, 7))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def test_mixed_in_odd_out(self):
        """Test a particular usage case."""
        image = np.zeros((6, 5))
        image[0:3, 0:3] = 1
        pupil_center_x = 1
        pupil_center_y = 1
        pixel_count_across = 7
        cropped_image = offcenter_crop(image, pixel_count_across,
                                       pupil_center_y, pupil_center_x)
        answer = pad_crop(np.ones((3, 3)), (7, 7))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def test_mixed_in_odd_out_change_center(self):
        """Test a particular usage case."""
        image = np.zeros((5, 5))
        image[0:3, 2::] = 1
        pupil_center_x = 3
        pupil_center_y = 1
        pixel_count_across = 7
        cropped_image = offcenter_crop(image, pixel_count_across,
                                       pupil_center_y, pupil_center_x)
        answer = pad_crop(np.ones((3, 3)), (7, 7))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def test_even_in_odd_out(self):
        """Test a particular usage case."""
        image = np.zeros((6, 10))
        image[3, 3] = 1
        pupil_center_x = 3
        pupil_center_y = 3
        pixel_count_across = 7
        cropped_image = offcenter_crop(image, pixel_count_across,
                                       pupil_center_y, pupil_center_x)
        answer = pad_crop(np.ones((1, 1)), (7, 7))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def test_even_in_even_out(self):
        """Test a particular usage case."""
        image = np.zeros((6, 6))
        image[0:2, 0:2] = 1
        pupil_center_x = 1
        pupil_center_y = 1
        pixel_count_across = 4
        cropped_image = offcenter_crop(image, pixel_count_across,
                                       pupil_center_y, pupil_center_x)
        answer = pad_crop(np.ones((2, 2)), (4, 4))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def test_even_in_even_out_extend_past_input_array(self):
        """Test a particular usage case."""
        image = np.zeros((6, 6))
        image[4::, 4::] = 1
        pupil_center_x = 5
        pupil_center_y = 5
        pixel_count_across = 4
        cropped_image = offcenter_crop(image, pixel_count_across,
                                       pupil_center_y, pupil_center_x)
        answer = pad_crop(np.ones((2, 2)), (4, 4))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)

    def fully_outside_input_array(self):
        """Test a particular usage case."""
        arrayIn = np.eye(4)
        cropped_image = offcenter_crop(arrayIn, 10, -20, 20)
        answer = np.zeros((10, 10))
        self.assertTrue(np.sum(np.abs(cropped_image - answer)) == 0)


if __name__ == '__main__':
    unittest.main()
