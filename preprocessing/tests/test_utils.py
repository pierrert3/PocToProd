import unittest
import pandas as pd
from unittest.mock import MagicMock

from preprocessing.preprocessing import utils


class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # We assert that our number of train batches will return (100*train_ratio)/20 = 80/20 = 4
        self.assertEqual(base._get_num_train_batches(), 4)

    def test__get_num_test_batches(self):
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # We assert that our number of test batches will return (100*(1-train_ratio))/20 = 20/20 = 1
        self.assertEqual(base._get_num_test_batches(), 1)

    def test_get_index_to_label_map(self):
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value ['Label0','Label1','Label2']
        # pd.unique() should return an array like value
        base._get_label_list = MagicMock(return_value=['Label0', 'Label1', 'Label2'])
        # We assert that our index to label should be {0:'Label0',1:'Label1',2:'Label2'}
        self.assertEqual(base.get_index_to_label_map(), {0: 'Label0', 1: 'Label1', 2: 'Label2'})

    def test_index_to_label_and_label_to_index_are_identity(self):
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we assert that mock _get_num_samples to return the value ['Label0','Label1','Label2']
        # pd.unique() should return an array like value
        base._get_label_list = MagicMock(return_value=['Label0', 'Label1', 'Label2'])
        # We assert that our get_index_to_label_map().keys() =get_label_to_index_map().values()
        # and get_index_to_label_map().values() = get_label_to_index_map().keys()
        self.assertEqual(list(base.get_index_to_label_map().keys()), list(base.get_label_to_index_map().values()))
        self.assertEqual(list(base.get_index_to_label_map().values()), list(base.get_label_to_index_map().keys()))

    def test_to_indexes(self):
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value ['Label0','Label1','Label2']
        # pd.unique() should return an array like value
        base._get_label_list = MagicMock(return_value=['Label0', 'Label1', 'Label2'])
        # We assert that if we send ['Label0','Label2'] to the to_indexes function, it will return [0,2]
        self.assertEqual(base.to_indexes(['Label0', 'Label2']), [0, 2])


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_c', 'tag_c'],
            'tag_id': [1, 2, 1, 3, 3],
            'tag_position': [0, 1, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        local = utils.LocalTextCategorizationDataset("fake_path", 1, 0.5,1)
        # dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        self.assertEqual(local._get_num_samples(), 4)

    def test_get_train_batch_returns_expected_shape(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_c', 'tag_c'],
            'tag_id': [1, 2, 1, 3, 3],
            'tag_position': [0, 1, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        local = utils.LocalTextCategorizationDataset("fake_path", 2, 0.5, 1)
        next_x, next_y = local.get_train_batch()
        self.assertEqual(next_x.shape, (2,))
        self.assertEqual(next_y.shape, (2, 2))

    def test_get_test_batch_returns_expected_shape(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_c', 'tag_c'],
            'tag_id': [1, 2, 1, 3, 3],
            'tag_position': [0, 1, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        local = utils.LocalTextCategorizationDataset("fake_path", 2, 0.5, 1)
        next_x, next_y = local.get_test_batch()
        self.assertEqual(next_x.shape, (2,))
        self.assertEqual(next_y.shape, (2, 2))

    def test_get_train_batch_raises_assertion_error(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2', 'id_3', 'id_4', 'id_5'],
            'tag_name': ['tag_a', 'tag_b', 'tag_a', 'tag_c', 'tag_c'],
            'tag_id': [1, 2, 1, 3, 3],
            'tag_position': [0, 1, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5']
        }))

        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv),
        # and we load dataset with an assert error (batch size too big, batch will be empty
        with self.assertRaises(AssertionError):
            local = utils.LocalTextCategorizationDataset("fake_path", 6, 0.5, 1)
