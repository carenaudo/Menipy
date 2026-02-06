# Docstring & Comment Remediation Plan

## Summary

This report identifies files with insufficient docstrings/comments and recommends fixes.
**Docstring coverage: 47.69% across the repo** (~1000 of 2165 functions/classes documented).

## Remediation Priority

- **HIGH**: Core modules (src/menipy/gui, src/menipy/cli, src/menipy/common, plugins, examples)
- **MEDIUM**: Pipeline modules, utility scripts, root-level helpers
- **LOW**: Tests, prototypes, playground code

## Files Requiring Remediation


### HIGH Priority

**1. plugins\auto_adaptive_edge.py**
   - **Effort**: 0/5 | **Coverage**: 85.7% (6/7)
   - **Issues**: low_docstring_coverage_85.7%
   - **Lines**: 252 | **TODO/FIXME**: 0

**2. plugins\bezier_edge.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 15 | **TODO/FIXME**: 0

**3. plugins\circle_edge.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (1/2)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 62 | **TODO/FIXME**: 0

**4. plugins\detect_apex.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/3)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 125 | **TODO/FIXME**: 0

**5. plugins\detect_drop.py**
   - **Effort**: 0/5 | **Coverage**: 40.0% (2/5)
   - **Issues**: low_docstring_coverage_40.0%
   - **Lines**: 277 | **TODO/FIXME**: 0

**6. plugins\detect_needle.py**
   - **Effort**: 0/5 | **Coverage**: 40.0% (2/5)
   - **Issues**: low_docstring_coverage_40.0%
   - **Lines**: 244 | **TODO/FIXME**: 0

**7. plugins\detect_roi.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/3)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 177 | **TODO/FIXME**: 0

**8. plugins\detect_substrate.py**
   - **Effort**: 0/5 | **Coverage**: 33.3% (1/3)
   - **Issues**: low_docstring_coverage_33.3%
   - **Lines**: 176 | **TODO/FIXME**: 0

**9. plugins\sine_edge.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (1/2)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 57 | **TODO/FIXME**: 0

**10. plugins\young_laplace_adsa.py**
   - **Effort**: 0/5 | **Coverage**: 42.9% (3/7)
   - **Issues**: low_docstring_coverage_42.9%
   - **Lines**: 214 | **TODO/FIXME**: 0

**11. src\menipy\common\plugins.py**
   - **Effort**: 0/5 | **Coverage**: 62.5% (5/8)
   - **Issues**: low_docstring_coverage_62.5%
   - **Lines**: 250 | **TODO/FIXME**: 0

**12. plugins\output_json.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 13 | **TODO/FIXME**: 0

**13. plugins\overlayer_simple.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 12 | **TODO/FIXME**: 0

**14. plugins\physics_dummy.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 11 | **TODO/FIXME**: 0

**15. plugins\preproc_blur.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 21 | **TODO/FIXME**: 0

**16. plugins\scaler_identity.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 12 | **TODO/FIXME**: 0

**17. plugins\validator_basic.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 12 | **TODO/FIXME**: 0

**18. src\menipy\gui\controllers\plugins_controller.py**
   - **Effort**: 1/5 | **Coverage**: 10.0% (1/10)
   - **Issues**: low_docstring_coverage_10.0%
   - **Lines**: 204 | **TODO/FIXME**: 0

**19. src\menipy\gui\viewmodels\plugins_vm.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/6)
   - **Issues**: low_docstring_coverage_0.0% | missing_type_hints
   - **Lines**: 33 | **TODO/FIXME**: 0

**20. plugins\edge_detectors.py**
   - **Effort**: 3/5 | **Coverage**: 30.8% (8/26)
   - **Issues**: low_docstring_coverage_30.8%
   - **Lines**: 542 | **TODO/FIXME**: 0

**21. tests\test_detection_plugins.py**
   - **Effort**: 3/5 | **Coverage**: 30.4% (7/23)
   - **Issues**: low_docstring_coverage_30.4%
   - **Lines**: 280 | **TODO/FIXME**: 0

**22. tests\test_preproc_plugins.py**
   - **Effort**: 3/5 | **Coverage**: 30.4% (7/23)
   - **Issues**: low_docstring_coverage_30.4%
   - **Lines**: 178 | **TODO/FIXME**: 0


### MEDIUM Priority

**23. scripts\add_gui_docstrings.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (1/2)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 99 | **TODO/FIXME**: 0

**24. scripts\add_stub_docstrings.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (1/2)
   - **Issues**: low_docstring_coverage_50.0% | todo_fixme_x5
   - **Lines**: 177 | **TODO/FIXME**: 5

**25. scripts\find_unreferenced.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 175 | **TODO/FIXME**: 0

**26. scripts\generate_docs.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/4)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 99 | **TODO/FIXME**: 0

**27. scripts\import_all_menipy.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 64 | **TODO/FIXME**: 0

**28. scripts\playground\droplet_analysis_skimage.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (6/6)
   - **Issues**: missing_type_hints
   - **Lines**: 268 | **TODO/FIXME**: 0

**29. scripts\playground\droplet_snake_analysis.py**
   - **Effort**: 0/5 | **Coverage**: 85.7% (6/7)
   - **Issues**: low_docstring_coverage_85.7% | missing_type_hints
   - **Lines**: 261 | **TODO/FIXME**: 0

**30. scripts\playground\sessile_calculations.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (2/4)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 369 | **TODO/FIXME**: 0

**31. scripts\playground\sessile_calculations2.py**
   - **Effort**: 0/5 | **Coverage**: 66.7% (6/9)
   - **Issues**: low_docstring_coverage_66.7% | missing_type_hints
   - **Lines**: 591 | **TODO/FIXME**: 0

**32. scripts\playground\sessile_calculations3.py**
   - **Effort**: 0/5 | **Coverage**: 77.8% (7/9)
   - **Issues**: low_docstring_coverage_77.8% | missing_type_hints
   - **Lines**: 621 | **TODO/FIXME**: 0

**33. scripts\playground\sessile_detection_method.py**
   - **Effort**: 0/5 | **Coverage**: 66.7% (2/3)
   - **Issues**: low_docstring_coverage_66.7%
   - **Lines**: 239 | **TODO/FIXME**: 0

**34. scripts\playground\sessile_detection_method2.py**
   - **Effort**: 0/5 | **Coverage**: 83.3% (5/6)
   - **Issues**: low_docstring_coverage_83.3% | missing_type_hints
   - **Lines**: 333 | **TODO/FIXME**: 0

**35. scripts\playground\sessile_preprocessing_tests.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (13/13)
   - **Issues**: missing_type_hints
   - **Lines**: 472 | **TODO/FIXME**: 0

**36. scripts\playground\synth_val.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (18/18)
   - **Issues**: missing_type_hints
   - **Lines**: 536 | **TODO/FIXME**: 0

**37. scripts\playground\test_real_images.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (2/4)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 156 | **TODO/FIXME**: 0

**38. src\menipy\common\acquisition.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (4/4)
   - **Issues**: todo_fixme_x1
   - **Lines**: 61 | **TODO/FIXME**: 1

**39. src\menipy\common\contour_smoothing.py**
   - **Effort**: 0/5 | **Coverage**: 40.0% (2/5)
   - **Issues**: low_docstring_coverage_40.0%
   - **Lines**: 359 | **TODO/FIXME**: 0

**40. src\menipy\common\detection_helpers.py**
   - **Effort**: 0/5 | **Coverage**: 25.0% (1/4)
   - **Issues**: low_docstring_coverage_25.0%
   - **Lines**: 217 | **TODO/FIXME**: 0

**41. src\menipy\common\edge_detection.py**
   - **Effort**: 0/5 | **Coverage**: 66.7% (4/6)
   - **Issues**: low_docstring_coverage_66.7%
   - **Lines**: 390 | **TODO/FIXME**: 0

**42. src\menipy\common\image_utils.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (1/2)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 60 | **TODO/FIXME**: 0

**43. src\menipy\common\metrics.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (1/2)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 42 | **TODO/FIXME**: 0

**44. src\menipy\common\optimization.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (1/1)
   - **Issues**: todo_fixme_x1
   - **Lines**: 15 | **TODO/FIXME**: 1

**45. src\menipy\common\overlay.py**
   - **Effort**: 0/5 | **Coverage**: 33.3% (2/6)
   - **Issues**: low_docstring_coverage_33.3%
   - **Lines**: 167 | **TODO/FIXME**: 0

**46. src\menipy\common\plugin_loader.py**
   - **Effort**: 0/5 | **Coverage**: 92.9% (13/14)
   - **Issues**: low_docstring_coverage_92.9%
   - **Lines**: 229 | **TODO/FIXME**: 0

**47. src\menipy\common\plugin_settings.py**
   - **Effort**: 0/5 | **Coverage**: 66.7% (2/3)
   - **Issues**: low_docstring_coverage_66.7%
   - **Lines**: 64 | **TODO/FIXME**: 0

**48. src\menipy\gui\adsa_app.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (1/2)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 70 | **TODO/FIXME**: 0

**49. src\menipy\gui\controllers\camera_manager.py**
   - **Effort**: 0/5 | **Coverage**: 83.3% (5/6)
   - **Issues**: low_docstring_coverage_83.3%
   - **Lines**: 130 | **TODO/FIXME**: 0

**50. src\menipy\gui\controllers\image_manager.py**
   - **Effort**: 0/5 | **Coverage**: 66.7% (4/6)
   - **Issues**: low_docstring_coverage_66.7%
   - **Lines**: 121 | **TODO/FIXME**: 0

**51. src\menipy\gui\controllers\layout_manager.py**
   - **Effort**: 0/5 | **Coverage**: 75.0% (3/4)
   - **Issues**: low_docstring_coverage_75.0%
   - **Lines**: 73 | **TODO/FIXME**: 0

**52. src\menipy\gui\controllers\overlay_manager.py**
   - **Effort**: 0/5 | **Coverage**: 57.1% (4/7)
   - **Issues**: low_docstring_coverage_57.1% | missing_type_hints
   - **Lines**: 199 | **TODO/FIXME**: 0

**53. src\menipy\gui\controllers\pipeline_ui_manager.py**
   - **Effort**: 0/5 | **Coverage**: 83.3% (10/12)
   - **Issues**: low_docstring_coverage_83.3%
   - **Lines**: 139 | **TODO/FIXME**: 0

**54. src\menipy\gui\dialogs\acquisition_config_dialog.py**
   - **Effort**: 0/5 | **Coverage**: 33.3% (1/3)
   - **Issues**: low_docstring_coverage_33.3%
   - **Lines**: 46 | **TODO/FIXME**: 0

**55. src\menipy\gui\dialogs\analysis_settings\captive_bubble_settings.py**
   - **Effort**: 0/5 | **Coverage**: 33.3% (1/3)
   - **Issues**: low_docstring_coverage_33.3%
   - **Lines**: 66 | **TODO/FIXME**: 0

**56. src\menipy\gui\dialogs\analysis_settings\pendant_settings.py**
   - **Effort**: 0/5 | **Coverage**: 33.3% (1/3)
   - **Issues**: low_docstring_coverage_33.3%
   - **Lines**: 118 | **TODO/FIXME**: 0

**57. src\menipy\gui\dialogs\analysis_settings\sessile_settings.py**
   - **Effort**: 0/5 | **Coverage**: 33.3% (1/3)
   - **Issues**: low_docstring_coverage_33.3%
   - **Lines**: 120 | **TODO/FIXME**: 0

**58. src\menipy\gui\dialogs\calibration_wizard_dialog.py**
   - **Effort**: 0/5 | **Coverage**: 91.7% (22/24)
   - **Issues**: low_docstring_coverage_91.7%
   - **Lines**: 817 | **TODO/FIXME**: 0

**59. src\menipy\gui\dialogs\physics_config_dialog.py**
   - **Effort**: 0/5 | **Coverage**: 75.0% (3/4)
   - **Issues**: low_docstring_coverage_75.0%
   - **Lines**: 90 | **TODO/FIXME**: 0

**60. src\menipy\gui\dialogs\plugin_config_dialog.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/4)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 141 | **TODO/FIXME**: 0

**61. src\menipy\gui\dialogs\utilities_dialog.py**
   - **Effort**: 0/5 | **Coverage**: 85.7% (6/7)
   - **Issues**: low_docstring_coverage_85.7% | missing_type_hints
   - **Lines**: 262 | **TODO/FIXME**: 0

**62. src\menipy\gui\main_controller.py**
   - **Effort**: 0/5 | **Coverage**: 83.3% (15/18)
   - **Issues**: low_docstring_coverage_83.3%
   - **Lines**: 394 | **TODO/FIXME**: 0

**63. src\menipy\gui\overlay.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (1/2)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 123 | **TODO/FIXME**: 0

**64. src\menipy\gui\panels\action_panel.py**
   - **Effort**: 0/5 | **Coverage**: 85.7% (6/7)
   - **Issues**: low_docstring_coverage_85.7% | missing_type_hints
   - **Lines**: 185 | **TODO/FIXME**: 0

**65. src\menipy\gui\panels\calibration_panel.py**
   - **Effort**: 0/5 | **Coverage**: 75.0% (6/8)
   - **Issues**: low_docstring_coverage_75.0%
   - **Lines**: 193 | **TODO/FIXME**: 0

**66. src\menipy\gui\panels\image_source_panel.py**
   - **Effort**: 0/5 | **Coverage**: 93.3% (14/15)
   - **Issues**: low_docstring_coverage_93.3% | todo_fixme_x1
   - **Lines**: 321 | **TODO/FIXME**: 1

**67. src\menipy\gui\panels\needle_calibration_panel.py**
   - **Effort**: 0/5 | **Coverage**: 92.3% (12/13)
   - **Issues**: low_docstring_coverage_92.3%
   - **Lines**: 240 | **TODO/FIXME**: 0

**68. src\menipy\gui\panels\parameters_panel.py**
   - **Effort**: 0/5 | **Coverage**: 77.8% (7/9)
   - **Issues**: low_docstring_coverage_77.8%
   - **Lines**: 308 | **TODO/FIXME**: 0

**69. src\menipy\gui\panels\tilt_stage_panel.py**
   - **Effort**: 0/5 | **Coverage**: 92.3% (12/13)
   - **Issues**: low_docstring_coverage_92.3%
   - **Lines**: 281 | **TODO/FIXME**: 0

**70. src\menipy\gui\resources\menipy_icons_rc.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/2)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 196 | **TODO/FIXME**: 0

**71. src\menipy\gui\services\image_convert.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 18 | **TODO/FIXME**: 0

**72. src\menipy\gui\services\settings_service.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/4)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 56 | **TODO/FIXME**: 0

**73. src\menipy\gui\views\base_experiment_window.py**
   - **Effort**: 0/5 | **Coverage**: 95.8% (23/24)
   - **Issues**: low_docstring_coverage_95.8%
   - **Lines**: 423 | **TODO/FIXME**: 0

**74. src\menipy\gui\views\experiment_selector.py**
   - **Effort**: 0/5 | **Coverage**: 81.2% (13/16)
   - **Issues**: low_docstring_coverage_81.2% | todo_fixme_x1 | missing_type_hints
   - **Lines**: 319 | **TODO/FIXME**: 1

**75. src\menipy\gui\widgets\experiment_card.py**
   - **Effort**: 0/5 | **Coverage**: 76.9% (10/13)
   - **Issues**: low_docstring_coverage_76.9%
   - **Lines**: 223 | **TODO/FIXME**: 0

**76. src\menipy\gui\widgets\interactive_image_viewer.py**
   - **Effort**: 0/5 | **Coverage**: 76.5% (13/17)
   - **Issues**: low_docstring_coverage_76.5%
   - **Lines**: 247 | **TODO/FIXME**: 0

**77. src\menipy\gui\widgets\measurements_table.py**
   - **Effort**: 0/5 | **Coverage**: 60.0% (3/5)
   - **Issues**: low_docstring_coverage_60.0%
   - **Lines**: 106 | **TODO/FIXME**: 0

**78. src\menipy\gui\widgets\notification.py**
   - **Effort**: 0/5 | **Coverage**: 71.4% (5/7)
   - **Issues**: low_docstring_coverage_71.4% | missing_type_hints
   - **Lines**: 146 | **TODO/FIXME**: 0

**79. src\menipy\gui\widgets\pendant_results_widget.py**
   - **Effort**: 0/5 | **Coverage**: 80.0% (8/10)
   - **Issues**: low_docstring_coverage_80.0%
   - **Lines**: 300 | **TODO/FIXME**: 0

**80. src\menipy\gui\widgets\quick_stats_widget.py**
   - **Effort**: 0/5 | **Coverage**: 88.9% (8/9)
   - **Issues**: low_docstring_coverage_88.9%
   - **Lines**: 233 | **TODO/FIXME**: 0

**81. src\menipy\gui\widgets\tilted_sessile_results_widget.py**
   - **Effort**: 0/5 | **Coverage**: 90.0% (9/10)
   - **Issues**: low_docstring_coverage_90.0%
   - **Lines**: 300 | **TODO/FIXME**: 0

**82. src\menipy\models\config.py**
   - **Effort**: 0/5 | **Coverage**: 78.6% (11/14)
   - **Issues**: low_docstring_coverage_78.6%
   - **Lines**: 313 | **TODO/FIXME**: 0

**83. src\menipy\models\context.py**
   - **Effort**: 0/5 | **Coverage**: 20.0% (1/5)
   - **Issues**: low_docstring_coverage_20.0%
   - **Lines**: 123 | **TODO/FIXME**: 0

**84. src\menipy\models\drop_extras.py**
   - **Effort**: 0/5 | **Coverage**: 83.3% (5/6)
   - **Issues**: low_docstring_coverage_83.3%
   - **Lines**: 63 | **TODO/FIXME**: 0

**85. src\menipy\models\fit.py**
   - **Effort**: 0/5 | **Coverage**: 83.3% (5/6)
   - **Issues**: low_docstring_coverage_83.3%
   - **Lines**: 85 | **TODO/FIXME**: 0

**86. src\menipy\models\frame.py**
   - **Effort**: 0/5 | **Coverage**: 60.0% (3/5)
   - **Issues**: low_docstring_coverage_60.0%
   - **Lines**: 79 | **TODO/FIXME**: 0

**87. src\menipy\models\geometry.py**
   - **Effort**: 0/5 | **Coverage**: 87.5% (7/8)
   - **Issues**: low_docstring_coverage_87.5%
   - **Lines**: 95 | **TODO/FIXME**: 0

**88. src\menipy\models\properties.py**
   - **Effort**: 0/5 | **Coverage**: 33.3% (1/3)
   - **Issues**: low_docstring_coverage_33.3%
   - **Lines**: 127 | **TODO/FIXME**: 0

**89. src\menipy\models\result.py**
   - **Effort**: 0/5 | **Coverage**: 80.0% (4/5)
   - **Issues**: low_docstring_coverage_80.0%
   - **Lines**: 93 | **TODO/FIXME**: 0

**90. src\menipy\models\results.py**
   - **Effort**: 0/5 | **Coverage**: 77.8% (7/9)
   - **Issues**: low_docstring_coverage_77.8%
   - **Lines**: 170 | **TODO/FIXME**: 0

**91. src\menipy\pipelines\capillary_rise\acquisition.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**92. src\menipy\pipelines\capillary_rise\edge_detection.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**93. src\menipy\pipelines\capillary_rise\geometry.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**94. src\menipy\pipelines\capillary_rise\optimization.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**95. src\menipy\pipelines\capillary_rise\outputs.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**96. src\menipy\pipelines\capillary_rise\overlay.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**97. src\menipy\pipelines\capillary_rise\physics.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**98. src\menipy\pipelines\capillary_rise\preprocessing.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**99. src\menipy\pipelines\capillary_rise\scaling.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**100. src\menipy\pipelines\capillary_rise\solver.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**101. src\menipy\pipelines\capillary_rise\validation.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**102. src\menipy\pipelines\captive_bubble\acquisition.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**103. src\menipy\pipelines\captive_bubble\edge_detection.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**104. src\menipy\pipelines\captive_bubble\geometry.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**105. src\menipy\pipelines\captive_bubble\optimization.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**106. src\menipy\pipelines\captive_bubble\outputs.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**107. src\menipy\pipelines\captive_bubble\overlay.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**108. src\menipy\pipelines\captive_bubble\physics.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**109. src\menipy\pipelines\captive_bubble\preprocessing.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**110. src\menipy\pipelines\captive_bubble\scaling.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**111. src\menipy\pipelines\captive_bubble\solver.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**112. src\menipy\pipelines\captive_bubble\validation.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**113. src\menipy\pipelines\oscillating\acquisition.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**114. src\menipy\pipelines\oscillating\edge_detection.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**115. src\menipy\pipelines\oscillating\geometry.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**116. src\menipy\pipelines\oscillating\optimization.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**117. src\menipy\pipelines\oscillating\outputs.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**118. src\menipy\pipelines\oscillating\overlay.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**119. src\menipy\pipelines\oscillating\physics.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**120. src\menipy\pipelines\oscillating\preprocessing.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**121. src\menipy\pipelines\oscillating\scaling.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**122. src\menipy\pipelines\oscillating\solver.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**123. src\menipy\pipelines\oscillating\validation.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**124. src\menipy\pipelines\pendant\acquisition.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**125. src\menipy\pipelines\pendant\drawing.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (1/1)
   - **Issues**: todo_fixme_x1
   - **Lines**: 27 | **TODO/FIXME**: 1

**126. src\menipy\pipelines\pendant\edge_detection.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**127. src\menipy\pipelines\pendant\geometry.py**
   - **Effort**: 0/5 | **Coverage**: 50.0% (2/4)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 62 | **TODO/FIXME**: 0

**128. src\menipy\pipelines\pendant\metrics.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 118 | **TODO/FIXME**: 0

**129. src\menipy\pipelines\pendant\optimization.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**130. src\menipy\pipelines\pendant\outputs.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 22 | **TODO/FIXME**: 0

**131. src\menipy\pipelines\pendant\overlay.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**132. src\menipy\pipelines\pendant\physics.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 17 | **TODO/FIXME**: 0

**133. src\menipy\pipelines\pendant\scaling.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 15 | **TODO/FIXME**: 0

**134. src\menipy\pipelines\pendant\solver.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**135. src\menipy\pipelines\pendant\stages.py**
   - **Effort**: 0/5 | **Coverage**: 66.7% (8/12)
   - **Issues**: low_docstring_coverage_66.7%
   - **Lines**: 295 | **TODO/FIXME**: 0

**136. src\menipy\pipelines\pendant\validation.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 19 | **TODO/FIXME**: 0

**137. src\menipy\pipelines\runner.py**
   - **Effort**: 0/5 | **Coverage**: 66.7% (2/3)
   - **Issues**: low_docstring_coverage_66.7%
   - **Lines**: 48 | **TODO/FIXME**: 0

**138. src\menipy\pipelines\sessile\acquisition.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**139. src\menipy\pipelines\sessile\drawing.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (1/1)
   - **Issues**: todo_fixme_x1
   - **Lines**: 27 | **TODO/FIXME**: 1

**140. src\menipy\pipelines\sessile\edge_detection.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**141. src\menipy\pipelines\sessile\metrics.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 238 | **TODO/FIXME**: 0

**142. src\menipy\pipelines\sessile\optimization.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**143. src\menipy\pipelines\sessile\outputs.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 24 | **TODO/FIXME**: 0

**144. src\menipy\pipelines\sessile\overlay.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**145. src\menipy\pipelines\sessile\physics.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 17 | **TODO/FIXME**: 0

**146. src\menipy\pipelines\sessile\scaling.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 18 | **TODO/FIXME**: 0

**147. src\menipy\pipelines\sessile\solver.py**
   - **Effort**: 0/5 | **Coverage**: N/A (0/0)
   - **Issues**: todo_fixme_x1
   - **Lines**: 14 | **TODO/FIXME**: 1

**148. src\menipy\pipelines\sessile\stages.py**
   - **Effort**: 0/5 | **Coverage**: 75.0% (9/12)
   - **Issues**: low_docstring_coverage_75.0%
   - **Lines**: 378 | **TODO/FIXME**: 0

**149. src\menipy\pipelines\sessile\validation.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 19 | **TODO/FIXME**: 0

**150. src\menipy\pipelines\utils.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 87 | **TODO/FIXME**: 0

**151. tests\test_sessile_geometry.py**
   - **Effort**: 0/5 | **Coverage**: 0.0% (0/4)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 148 | **TODO/FIXME**: 0

**152. tools\audit_docstrings.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (5/5)
   - **Issues**: todo_fixme_x35
   - **Lines**: 335 | **TODO/FIXME**: 35

**153. docs\conf.py**
   - **Effort**: 1/5 | **Coverage**: N/A (0/0)
   - **Issues**: missing_module_docstring | todo_fixme_x1
   - **Lines**: 10 | **TODO/FIXME**: 1

**154. pendant_detections.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 196 | **TODO/FIXME**: 0

**155. scripts\generate_legacy_map.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/5)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 69 | **TODO/FIXME**: 0

**156. scripts\playground\synth_gen.py**
   - **Effort**: 1/5 | **Coverage**: 63.6% (14/22)
   - **Issues**: low_docstring_coverage_63.6% | missing_type_hints
   - **Lines**: 917 | **TODO/FIXME**: 0

**157. scripts\playground\synth_scripts.py**
   - **Effort**: 1/5 | **Coverage**: N/A (0/0)
   - **Issues**: missing_module_docstring
   - **Lines**: 20 | **TODO/FIXME**: 0

**158. src\menipy\cli.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/8)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 320 | **TODO/FIXME**: 0

**159. src\menipy\cli\__init__.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/6)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 197 | **TODO/FIXME**: 0

**160. src\menipy\common\auto_calibrator.py**
   - **Effort**: 1/5 | **Coverage**: 72.2% (13/18)
   - **Issues**: low_docstring_coverage_72.2%
   - **Lines**: 813 | **TODO/FIXME**: 0

**161. src\menipy\common\geometry.py**
   - **Effort**: 1/5 | **Coverage**: 36.4% (4/11)
   - **Issues**: low_docstring_coverage_36.4%
   - **Lines**: 584 | **TODO/FIXME**: 0

**162. src\menipy\common\material_db.py**
   - **Effort**: 1/5 | **Coverage**: 38.5% (5/13)
   - **Issues**: low_docstring_coverage_38.5%
   - **Lines**: 269 | **TODO/FIXME**: 0

**163. src\menipy\common\preprocessing.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/5)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 244 | **TODO/FIXME**: 0

**164. src\menipy\common\solver.py**
   - **Effort**: 1/5 | **Coverage**: 14.3% (1/7)
   - **Issues**: low_docstring_coverage_14.3%
   - **Lines**: 175 | **TODO/FIXME**: 0

**165. src\menipy\gui\app.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/7)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 144 | **TODO/FIXME**: 0

**166. src\menipy\gui\components\plugin_settings_widget.py**
   - **Effort**: 1/5 | **Coverage**: 50.0% (5/10)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 272 | **TODO/FIXME**: 0

**167. src\menipy\gui\controllers\dialog_coordinator.py**
   - **Effort**: 1/5 | **Coverage**: 76.2% (16/21)
   - **Issues**: low_docstring_coverage_76.2%
   - **Lines**: 415 | **TODO/FIXME**: 0

**168. src\menipy\gui\controllers\edge_detection_controller.py**
   - **Effort**: 1/5 | **Coverage**: 22.2% (2/9)
   - **Issues**: low_docstring_coverage_22.2%
   - **Lines**: 121 | **TODO/FIXME**: 0

**169. src\menipy\gui\controllers\sop_controller.py**
   - **Effort**: 1/5 | **Coverage**: 10.0% (1/10)
   - **Issues**: low_docstring_coverage_10.0%
   - **Lines**: 203 | **TODO/FIXME**: 0

**170. src\menipy\gui\dialogs\help_dialog.py**
   - **Effort**: 1/5 | **Coverage**: 16.7% (1/6)
   - **Issues**: low_docstring_coverage_16.7%
   - **Lines**: 121 | **TODO/FIXME**: 0

**171. src\menipy\gui\dialogs\plugin_manager_dialog.py**
   - **Effort**: 1/5 | **Coverage**: 20.0% (2/10)
   - **Issues**: low_docstring_coverage_20.0%
   - **Lines**: 284 | **TODO/FIXME**: 0

**172. src\menipy\gui\dialogs\settings_dialog.py**
   - **Effort**: 1/5 | **Coverage**: 22.2% (2/9)
   - **Issues**: low_docstring_coverage_22.2%
   - **Lines**: 157 | **TODO/FIXME**: 0

**173. src\menipy\gui\logging_bridge.py**
   - **Effort**: 1/5 | **Coverage**: 28.6% (2/7)
   - **Issues**: low_docstring_coverage_28.6%
   - **Lines**: 86 | **TODO/FIXME**: 0

**174. src\menipy\gui\mainwindow.py**
   - **Effort**: 1/5 | **Coverage**: 25.0% (2/8)
   - **Issues**: low_docstring_coverage_25.0% | missing_type_hints
   - **Lines**: 164 | **TODO/FIXME**: 0

**175. src\menipy\gui\panels\results_panel.py**
   - **Effort**: 1/5 | **Coverage**: 62.5% (10/16)
   - **Issues**: low_docstring_coverage_62.5%
   - **Lines**: 378 | **TODO/FIXME**: 0

**176. src\menipy\gui\services\pipeline_runner.py**
   - **Effort**: 1/5 | **Coverage**: 11.1% (1/9)
   - **Issues**: low_docstring_coverage_11.1%
   - **Lines**: 145 | **TODO/FIXME**: 0

**177. src\menipy\gui\services\plugin_service.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/6)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 28 | **TODO/FIXME**: 0

**178. src\menipy\gui\viewmodels\run_vm.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/5)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 73 | **TODO/FIXME**: 0

**179. src\menipy\gui\views\step_item_widget.py**
   - **Effort**: 1/5 | **Coverage**: 14.3% (1/7)
   - **Issues**: low_docstring_coverage_14.3%
   - **Lines**: 157 | **TODO/FIXME**: 0

**180. src\menipy\models\datatypes.py**
   - **Effort**: 1/5 | **Coverage**: 40.0% (4/10)
   - **Issues**: low_docstring_coverage_40.0%
   - **Lines**: 127 | **TODO/FIXME**: 0

**181. src\menipy\models\state.py**
   - **Effort**: 1/5 | **Coverage**: 37.5% (3/8)
   - **Issues**: low_docstring_coverage_37.5%
   - **Lines**: 94 | **TODO/FIXME**: 0

**182. src\menipy\pipelines\base.py**
   - **Effort**: 1/5 | **Coverage**: 81.5% (22/27)
   - **Issues**: low_docstring_coverage_81.5%
   - **Lines**: 522 | **TODO/FIXME**: 0

**183. src\menipy\pipelines\capillary_rise\stages.py**
   - **Effort**: 1/5 | **Coverage**: 50.0% (5/10)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 151 | **TODO/FIXME**: 0

**184. src\menipy\pipelines\captive_bubble\stages.py**
   - **Effort**: 1/5 | **Coverage**: 50.0% (5/10)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 198 | **TODO/FIXME**: 0

**185. src\menipy\pipelines\oscillating\stages.py**
   - **Effort**: 1/5 | **Coverage**: 61.5% (8/13)
   - **Issues**: low_docstring_coverage_61.5%
   - **Lines**: 258 | **TODO/FIXME**: 0

**186. src\menipy\pipelines\pendant\preprocessing.py**
   - **Effort**: 1/5 | **Coverage**: 12.5% (1/8)
   - **Issues**: low_docstring_coverage_12.5%
   - **Lines**: 103 | **TODO/FIXME**: 0

**187. src\menipy\pipelines\sessile\geometry.py**
   - **Effort**: 1/5 | **Coverage**: 28.6% (2/7)
   - **Issues**: low_docstring_coverage_28.6%
   - **Lines**: 175 | **TODO/FIXME**: 0

**188. src\menipy\pipelines\sessile\preprocessing.py**
   - **Effort**: 1/5 | **Coverage**: 12.5% (1/8)
   - **Issues**: low_docstring_coverage_12.5%
   - **Lines**: 104 | **TODO/FIXME**: 0

**189. tools\check_indent.py**
   - **Effort**: 1/5 | **Coverage**: N/A (0/0)
   - **Issues**: missing_module_docstring
   - **Lines**: 17 | **TODO/FIXME**: 0

**190. tools\inspect_ui.py**
   - **Effort**: 1/5 | **Coverage**: N/A (0/0)
   - **Issues**: missing_module_docstring
   - **Lines**: 50 | **TODO/FIXME**: 0

**191. tools\migrate_datatypes_imports.py**
   - **Effort**: 1/5 | **Coverage**: 16.7% (1/6)
   - **Issues**: low_docstring_coverage_16.7%
   - **Lines**: 278 | **TODO/FIXME**: 0

**192. scripts\playground\prueba.py**
   - **Effort**: 2/5 | **Coverage**: 0.0% (0/3)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0% | large_commented_blocks
   - **Lines**: 377 | **TODO/FIXME**: 0

**193. scripts\playground\sessile_detections.py**
   - **Effort**: 2/5 | **Coverage**: 40.0% (8/20)
   - **Issues**: low_docstring_coverage_40.0%
   - **Lines**: 1577 | **TODO/FIXME**: 0

**194. src\menipy\common\plugin_db.py**
   - **Effort**: 2/5 | **Coverage**: 13.3% (2/15)
   - **Issues**: low_docstring_coverage_13.3%
   - **Lines**: 278 | **TODO/FIXME**: 0

**195. src\menipy\gui\controllers\pipeline_controller.py**
   - **Effort**: 2/5 | **Coverage**: 36.4% (8/22)
   - **Issues**: low_docstring_coverage_36.4%
   - **Lines**: 783 | **TODO/FIXME**: 0

**196. src\menipy\gui\controllers\preprocessing_controller.py**
   - **Effort**: 2/5 | **Coverage**: 6.7% (1/15)
   - **Issues**: low_docstring_coverage_6.7%
   - **Lines**: 226 | **TODO/FIXME**: 0

**197. src\menipy\gui\dialogs\calibration_wizard.py**
   - **Effort**: 2/5 | **Coverage**: 25.0% (4/16)
   - **Issues**: low_docstring_coverage_25.0%
   - **Lines**: 384 | **TODO/FIXME**: 0

**198. src\menipy\gui\dialogs\geometry_config_dialog.py**
   - **Effort**: 2/5 | **Coverage**: 16.7% (2/12)
   - **Issues**: low_docstring_coverage_16.7%
   - **Lines**: 266 | **TODO/FIXME**: 0

**199. src\menipy\gui\dialogs\material_dialog.py**
   - **Effort**: 2/5 | **Coverage**: 50.0% (10/20)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 460 | **TODO/FIXME**: 0

**200. src\menipy\gui\dialogs\overlay_config_dialog.py**
   - **Effort**: 2/5 | **Coverage**: 12.5% (2/16)
   - **Issues**: low_docstring_coverage_12.5%
   - **Lines**: 387 | **TODO/FIXME**: 0

**201. src\menipy\gui\helpers\image_marking.py**
   - **Effort**: 2/5 | **Coverage**: 9.1% (1/11)
   - **Issues**: low_docstring_coverage_9.1%
   - **Lines**: 211 | **TODO/FIXME**: 0

**202. src\menipy\gui\main_window.py**
   - **Effort**: 2/5 | **Coverage**: 12.5% (2/16)
   - **Issues**: low_docstring_coverage_12.5%
   - **Lines**: 463 | **TODO/FIXME**: 0

**203. src\menipy\gui\services\camera_service.py**
   - **Effort**: 2/5 | **Coverage**: 16.7% (2/12)
   - **Issues**: low_docstring_coverage_16.7%
   - **Lines**: 145 | **TODO/FIXME**: 0

**204. src\menipy\gui\services\sop_service.py**
   - **Effort**: 2/5 | **Coverage**: 0.0% (0/12)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 84 | **TODO/FIXME**: 0

**205. src\menipy\gui\views\tilted_sessile_window.py**
   - **Effort**: 2/5 | **Coverage**: 53.3% (16/30)
   - **Issues**: low_docstring_coverage_53.3%
   - **Lines**: 374 | **TODO/FIXME**: 0

**206. src\menipy\gui\views\ui_main_window.py**
   - **Effort**: 2/5 | **Coverage**: 0.0% (0/3)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0% | large_commented_blocks
   - **Lines**: 339 | **TODO/FIXME**: 0

**207. src\menipy\gui\dialogs\analysis_settings_dialog.py**
   - **Effort**: 3/5 | **Coverage**: 42.3% (11/26)
   - **Issues**: low_docstring_coverage_42.3%
   - **Lines**: 528 | **TODO/FIXME**: 0

**208. src\menipy\gui\dialogs\detector_test_dialog.py**
   - **Effort**: 3/5 | **Coverage**: 11.8% (2/17)
   - **Issues**: low_docstring_coverage_11.8% | missing_type_hints
   - **Lines**: 383 | **TODO/FIXME**: 0

**209. src\menipy\gui\dialogs\edge_detection_config_dialog.py**
   - **Effort**: 3/5 | **Coverage**: 21.7% (5/23)
   - **Issues**: low_docstring_coverage_21.7%
   - **Lines**: 624 | **TODO/FIXME**: 0

**210. src\menipy\gui\views\sessile_drop_window.py**
   - **Effort**: 3/5 | **Coverage**: 63.4% (26/41)
   - **Issues**: low_docstring_coverage_63.4%
   - **Lines**: 857 | **TODO/FIXME**: 0

**211. scripts\test_sessile_qt.py**
   - **Effort**: 4/5 | **Coverage**: 35.3% (12/34)
   - **Issues**: low_docstring_coverage_35.3%
   - **Lines**: 1679 | **TODO/FIXME**: 0

**212. src\menipy\common\preprocessing_helpers.py**
   - **Effort**: 4/5 | **Coverage**: 33.3% (11/33)
   - **Issues**: low_docstring_coverage_33.3%
   - **Lines**: 673 | **TODO/FIXME**: 0

**213. src\menipy\gui\panels\preview_panel.py**
   - **Effort**: 4/5 | **Coverage**: 13.0% (3/23)
   - **Issues**: low_docstring_coverage_13.0%
   - **Lines**: 260 | **TODO/FIXME**: 0

**214. src\menipy\gui\panels\setup_panel.py**
   - **Effort**: 4/5 | **Coverage**: 20.7% (6/29)
   - **Issues**: low_docstring_coverage_20.7%
   - **Lines**: 722 | **TODO/FIXME**: 0

**215. scripts\playground\prueba2.py**
   - **Effort**: 5/5 | **Coverage**: 3.1% (1/32)
   - **Issues**: low_docstring_coverage_3.1%
   - **Lines**: 929 | **TODO/FIXME**: 0

**216. src\menipy\common\registry.py**
   - **Effort**: 5/5 | **Coverage**: 16.7% (5/30)
   - **Issues**: low_docstring_coverage_16.7%
   - **Lines**: 168 | **TODO/FIXME**: 0

**217. src\menipy\gui\controllers\setup_panel_controller.py**
   - **Effort**: 5/5 | **Coverage**: 19.4% (6/31)
   - **Issues**: low_docstring_coverage_19.4%
   - **Lines**: 573 | **TODO/FIXME**: 0

**218. src\menipy\gui\dialogs\preprocessing_config_dialog.py**
   - **Effort**: 5/5 | **Coverage**: 3.7% (1/27)
   - **Issues**: low_docstring_coverage_3.7%
   - **Lines**: 706 | **TODO/FIXME**: 0

**219. src\menipy\gui\views\adsa_main_window.py**
   - **Effort**: 5/5 | **Coverage**: 57.6% (34/59)
   - **Issues**: low_docstring_coverage_57.6% | todo_fixme_x2
   - **Lines**: 979 | **TODO/FIXME**: 2

**220. src\menipy\gui\views\image_view.py**
   - **Effort**: 5/5 | **Coverage**: 9.7% (3/31)
   - **Issues**: low_docstring_coverage_9.7%
   - **Lines**: 649 | **TODO/FIXME**: 0

**221. src\menipy\gui\views\pendant_drop_window.py**
   - **Effort**: 5/5 | **Coverage**: 41.1% (23/56)
   - **Issues**: low_docstring_coverage_41.1%
   - **Lines**: 862 | **TODO/FIXME**: 0


### LOW Priority

**222. pruebas\droplet_analysis.py**
   - **Effort**: 0/5 | **Coverage**: 78.6% (11/14)
   - **Issues**: low_docstring_coverage_78.6% | missing_type_hints
   - **Lines**: 507 | **TODO/FIXME**: 0

**223. pruebas\droplet_analysis_v2.py**
   - **Effort**: 0/5 | **Coverage**: 84.2% (16/19)
   - **Issues**: low_docstring_coverage_84.2% | missing_type_hints
   - **Lines**: 900 | **TODO/FIXME**: 0

**224. tests\test_auto_calibrator.py**
   - **Effort**: 0/5 | **Coverage**: 93.5% (29/31)
   - **Issues**: low_docstring_coverage_93.5%
   - **Lines**: 388 | **TODO/FIXME**: 0

**225. tests\test_contour_smoothing.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (20/20)
   - **Issues**: missing_type_hints
   - **Lines**: 232 | **TODO/FIXME**: 0

**226. tests\test_project_loading.py**
   - **Effort**: 0/5 | **Coverage**: 71.4% (5/7)
   - **Issues**: low_docstring_coverage_71.4% | missing_type_hints
   - **Lines**: 108 | **TODO/FIXME**: 0

**227. tests\test_sessile_auto_detection.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (11/11)
   - **Issues**: missing_type_hints
   - **Lines**: 155 | **TODO/FIXME**: 0

**228. tests\test_sessile_contact_angles.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (8/8)
   - **Issues**: missing_type_hints
   - **Lines**: 213 | **TODO/FIXME**: 0

**229. tests\test_smoke_controller_flows.py**
   - **Effort**: 0/5 | **Coverage**: 100.0% (22/22)
   - **Issues**: missing_type_hints
   - **Lines**: 587 | **TODO/FIXME**: 0

**230. pruebas\data\generate_data.py**
   - **Effort**: 1/5 | **Coverage**: N/A (0/0)
   - **Issues**: missing_module_docstring
   - **Lines**: 36 | **TODO/FIXME**: 0

**231. pruebas\drop_analysis_script.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 172 | **TODO/FIXME**: 0

**232. pruebas\drop_analysis_script_v3.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 214 | **TODO/FIXME**: 0

**233. pruebas\script.py**
   - **Effort**: 1/5 | **Coverage**: N/A (0/0)
   - **Issues**: missing_module_docstring
   - **Lines**: 178 | **TODO/FIXME**: 0

**234. pruebas\test_geometry_helper_run.py**
   - **Effort**: 1/5 | **Coverage**: N/A (0/0)
   - **Issues**: missing_module_docstring
   - **Lines**: 25 | **TODO/FIXME**: 0

**235. tests\test_common_geometry_contact.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 30 | **TODO/FIXME**: 0

**236. tests\test_drop_extras.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/3)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 32 | **TODO/FIXME**: 0

**237. tests\test_gui_resources.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/3)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 37 | **TODO/FIXME**: 0

**238. tests\test_import_map.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/5)
   - **Issues**: low_docstring_coverage_0.0%
   - **Lines**: 138 | **TODO/FIXME**: 0

**239. tests\test_models.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/4)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 45 | **TODO/FIXME**: 0

**240. tests\test_overlay_dialog.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/3)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 31 | **TODO/FIXME**: 0

**241. tests\test_plugin_config_dialog.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/3)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 39 | **TODO/FIXME**: 0

**242. tests\test_preprocessing_compose.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/1)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 21 | **TODO/FIXME**: 0

**243. tests\test_surface_tension.py**
   - **Effort**: 1/5 | **Coverage**: 0.0% (0/3)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 35 | **TODO/FIXME**: 0

**244. tests\test_alt_workflow.py**
   - **Effort**: 2/5 | **Coverage**: 0.0% (0/5)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 71 | **TODO/FIXME**: 0

**245. tests\test_pendant_minimal_pipeline.py**
   - **Effort**: 2/5 | **Coverage**: 0.0% (0/5)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 55 | **TODO/FIXME**: 0

**246. tests\test_preprocessing_helpers.py**
   - **Effort**: 2/5 | **Coverage**: 0.0% (0/8)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 129 | **TODO/FIXME**: 0

**247. tests\test_sessile_minimal_pipeline.py**
   - **Effort**: 2/5 | **Coverage**: 0.0% (0/5)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 66 | **TODO/FIXME**: 0

**248. tests\test_edge_detectors_plugin.py**
   - **Effort**: 3/5 | **Coverage**: 39.3% (11/28)
   - **Issues**: low_docstring_coverage_39.3%
   - **Lines**: 292 | **TODO/FIXME**: 0

**249. tests\test_setup_panel.py**
   - **Effort**: 3/5 | **Coverage**: 0.0% (0/13)
   - **Issues**: missing_module_docstring | low_docstring_coverage_0.0%
   - **Lines**: 182 | **TODO/FIXME**: 0

**250. pruebas\droplet_analysis_v3.py**
   - **Effort**: 4/5 | **Coverage**: 50.0% (22/44)
   - **Issues**: low_docstring_coverage_50.0%
   - **Lines**: 1566 | **TODO/FIXME**: 0

**251. pruebas\sessile_analysis_v1.py**
   - **Effort**: 5/5 | **Coverage**: 45.3% (29/64)
   - **Issues**: low_docstring_coverage_45.3%
   - **Lines**: 1767 | **TODO/FIXME**: 0


## Statistics

- **HIGH priority files**: 22
- **MEDIUM priority files**: 199
- **LOW priority files**: 30
- **Total remediation candidates**: 251
- **Top issue**: Missing docstrings on functions/classes
