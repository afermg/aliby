import pytest
pytest.mark.skip("all tests still WIP")


from core.core import PersistentDict

# Todo: temporary file needed
class TestPersistentDict:
    @pytest.fixture(autouse=True, scope='class')
    def _get_json_file(self, tmp_path):
        self._filename = tmp_path / 'persistent_dict.json'

    def test_persistent_dict(self):
        p = PersistentDict(self._filename)
        p['hello/from/the/other/side'] = "adele"
        p['hello/how/you/doing'] = 'lionel'
        # Todo: run checks


# Todo: data needed - small experiment
class TestExperiment:
    def test_shape(self):
        pass
    def test_positions(self):
        pass
    def test_channels(self):
        pass
    def test_hypercube(self):
        pass

# Todo: data needed - a dummy OMERO server
class TestConnection:
    def test_dataset(self):
        pass
    def test_image(self):
        pass

# Todo data needed - a position
class TestTimelapse:
    def test_id(self):
        pass
    def test_name(self):
        pass
    def test_size_z(self):
        pass
    def test_size_c(self):
        pass
    def test_size_t(self):
        pass
    def test_size_x(self):
        pass
    def test_size_y(self):
        pass
    def test_channels(self):
        pass
    def test_channel_index(self):
        pass

# Todo: data needed image and template
class TestTrapUtils:
    def test_trap_locations(self):
        pass
    def test_tile_shape(self):
        pass
    def test_get_tile(self):
        pass
    def test_centre(self):
        pass

# Todo: data needed - a functional experiment object
class TestTiler:
    def test_n_timepoints(self):
        pass
    def test_n_traps(self):
        pass
    def test_get_trap_timelapse(self):
        pass
    def test_get_trap_timepoints(self):
        pass

# Todo: data needed - a functional tiler object
# Todo: running server needed
class TestBabyClient:
    def test_get_new_session(self):
        pass
    def test_queue_image(self):
        pass
    def test_get_segmentation(self):
        pass

# Todo: data needed - a functional tiler object
class TestBabyRunner:
    def test_model_choice(self):
        pass
    def test_properties(self):
        pass
    def test_segment(self):
        pass

