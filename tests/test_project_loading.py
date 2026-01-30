
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
from PySide6.QtCore import QSettings
from menipy.gui.views.adsa_main_window import ADSAMainWindow
from menipy.gui import theme

@pytest.fixture
def mock_settings():
    """Mock QSettings to avoid touching real registry/file."""
    settings = MagicMock(spec=QSettings)
    # Use a real dict to store settings
    settings.store = {}

    def get_val(key, default=None):
        return settings.store.get(key, default)

    def set_val(key, value):
        settings.store[key] = value

    settings.value.side_effect = get_val
    settings.setValue.side_effect = set_val
    return settings

@pytest.fixture
def window(mock_settings, qtbot):
    """Create ADSAMainWindow with mocked settings."""
    # Patch QSettings constructor to return our mock
    with patch("PySide6.QtCore.QSettings", return_value=mock_settings):
        # We also need to mock internal components that might require GUI/Hardware
        with patch("menipy.gui.views.adsa_main_window.QStatusBar"):
            win = ADSAMainWindow()
            win._settings_store = mock_settings # Ensure it uses our instance
            qtbot.addWidget(win)
            return win

def test_record_recent_adds_item(window, mock_settings):
    """Test that _record_recent adds items correctly."""
    # Initial state
    mock_settings.store["recent_projects"] = "[]"
    
    window._record_recent(
        experiment_type="sessile",
        title="Test Project",
        path="/path/to/project.adsa"
    )
    
    # Check stored value
    stored_json = mock_settings.store["recent_projects"]
    stored_list = json.loads(stored_json)
    
    assert len(stored_list) == 1
    assert stored_list[0]["filename"] == "Test Project"
    assert stored_list[0]["path"] == "/path/to/project.adsa"

def test_record_recent_deduplication(window, mock_settings):
    """Test that duplicate projects are moved to top."""
    initial = [
        {"filename": "Old", "experiment_type": "sessile", "date_str": "2023", "path": "/p1"},
        {"filename": "Test Project", "experiment_type": "sessile", "date_str": "2023", "path": "/p2"}
    ]
    mock_settings.store["recent_projects"] = json.dumps(initial)
    
    window._record_recent(
        experiment_type="sessile",
        title="Test Project", # Same as existing
        path="/p2"
    )
    
    stored_json = mock_settings.store["recent_projects"]
    stored_list = json.loads(stored_json)
    
    # Should still be 2 items, but "Test Project" is now at index 0
    assert len(stored_list) == 2
    assert stored_list[0]["filename"] == "Test Project"
    assert stored_list[1]["filename"] == "Old"

def test_open_project_file_success(window):
    """Test opening a valid project file."""
    
    project_data = {
        "experiment_type": theme.EXPERIMENT_SESSILE,
        "image_path": "/path/to/image.png",
        "analysis_settings": {"some": "settings"}
    }
    
    mock_file_content = json.dumps(project_data)
    
    # Mock show_experiment_window and the target window
    # We mock show_experiment_window on the instance
    window.show_experiment_window = MagicMock()
    
    # We also need window.get_current_experiment_window to return something
    mock_exp_window = MagicMock()
    window.get_current_experiment_window = MagicMock(return_value=mock_exp_window)
    
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        with patch("os.path.exists", return_value=True):
             window._open_project_file("/path/to/project.adsa")
             
             # Assertions
             window.show_experiment_window.assert_called_with(theme.EXPERIMENT_SESSILE)
             
             # Check that load_image was called on the returned window (detected via logic)
             # _open_project_file checks for load_image or _image_source_panel
             mock_exp_window.load_image.assert_called_with("/path/to/image.png")
