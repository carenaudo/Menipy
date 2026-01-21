
try:
    print("Importing pipeline_controller...")
    from menipy.gui.controllers import pipeline_controller
    print("Success pipeline_controller")
    
    print("Importing main_controller...")
    from menipy.gui import main_controller
    print("Success main_controller")

    print("Importing setup_panel_controller...")
    from menipy.gui.controllers import setup_panel_controller
    print("Success setup_panel_controller")

except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
