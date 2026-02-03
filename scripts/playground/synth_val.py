"""
ASDA Validation Framework
Tests your ASDA implementation against synthetic images with known ground truth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json


class ValidationMetrics:
    """Calculate validation metrics comparing results to ground truth"""

    @staticmethod
    def absolute_error(measured, true):
        """Absolute error"""
        return abs(measured - true)

    @staticmethod
    def relative_error(measured, true):
        """Relative error in percent"""
        if abs(true) < 1e-10:
            return np.nan
        return 100 * abs(measured - true) / abs(true)

    @staticmethod
    def mean_absolute_error(measured_list, true_list):
        """Mean absolute error"""
        return np.mean([abs(m - t) for m, t in zip(measured_list, true_list)])

    @staticmethod
    def rmse(measured_list, true_list):
        """Root mean square error"""
        return np.sqrt(
            np.mean([(m - t) ** 2 for m, t in zip(measured_list, true_list)])
        )

    @staticmethod
    def calculate_all_metrics(measured, true, metric_name):
        """Calculate all relevant metrics"""
        abs_err = ValidationMetrics.absolute_error(measured, true)
        rel_err = ValidationMetrics.relative_error(measured, true)

        return {
            f"{metric_name}_measured": measured,
            f"{metric_name}_true": true,
            f"{metric_name}_abs_error": abs_err,
            f"{metric_name}_rel_error": rel_err,
        }


class ASDATester:
    """Test ASDA analyzer against synthetic datasets"""

    def __init__(self, analyzer_class, calibration=20.0):
        """
        Parameters:
        -----------
        analyzer_class : class
            Your analyzer class (e.g., SessileDropAnalyzer)
        calibration : float
            Calibration in px/mm
        """
        self.analyzer_class = analyzer_class
        self.calibration = calibration
        self.results = []

    def test_single_image(self, image, ground_truth, verbose=True):
        """
        Test analyzer on single image

        Returns:
        --------
        metrics : dict
            Validation metrics
        """
        try:
            # Initialize analyzer
            if ground_truth.analysis_type == "sessile":
                analyzer = self.analyzer_class(calibration_px_per_mm=self.calibration)
            elif ground_truth.analysis_type == "pendant":
                analyzer = self.analyzer_class(
                    calibration_px_per_mm=self.calibration, density_diff=1000.0
                )
            else:
                analyzer = self.analyzer_class(calibration_px_per_mm=self.calibration)

            # Analyze (you'll need to implement analyze_from_array method)
            # This is a placeholder - adapt to your actual implementation
            session = analyzer.analyze_from_array(image)

            # Extract metrics based on analysis type
            metrics = {"analysis_type": ground_truth.analysis_type}

            if ground_truth.analysis_type == "sessile":
                metrics.update(
                    ValidationMetrics.calculate_all_metrics(
                        session.results["contact_angle_mean"],
                        ground_truth.contact_angle_deg,
                        "contact_angle",
                    )
                )

                if ground_truth.volume_uL:
                    metrics.update(
                        ValidationMetrics.calculate_all_metrics(
                            session.results["volume_uL"],
                            ground_truth.volume_uL,
                            "volume",
                        )
                    )

                if ground_truth.contact_diameter_mm:
                    metrics.update(
                        ValidationMetrics.calculate_all_metrics(
                            session.results["contact_diameter_mm"],
                            ground_truth.contact_diameter_mm,
                            "contact_diameter",
                        )
                    )

            elif ground_truth.analysis_type == "pendant":
                metrics.update(
                    ValidationMetrics.calculate_all_metrics(
                        session.results["surface_tension_mN_m"],
                        ground_truth.surface_tension_mN_m,
                        "surface_tension",
                    )
                )

                if ground_truth.bond_number:
                    metrics.update(
                        ValidationMetrics.calculate_all_metrics(
                            session.results["bond_number"],
                            ground_truth.bond_number,
                            "bond_number",
                        )
                    )

            elif ground_truth.analysis_type == "capillary":
                metrics.update(
                    ValidationMetrics.calculate_all_metrics(
                        session.results["rise_height_mm"],
                        ground_truth.rise_height_mm,
                        "rise_height",
                    )
                )

                metrics.update(
                    ValidationMetrics.calculate_all_metrics(
                        session.results["contact_angle_deg"],
                        ground_truth.contact_angle_deg,
                        "contact_angle",
                    )
                )

            elif ground_truth.analysis_type == "captive":
                metrics.update(
                    ValidationMetrics.calculate_all_metrics(
                        session.results["contact_angle_mean"],
                        ground_truth.contact_angle_deg,
                        "contact_angle",
                    )
                )

            # Add validation score
            metrics["validation_score"] = session.results.get(
                "validation_score", np.nan
            )
            metrics["success"] = True

            if verbose:
                print("✓ Analysis successful")
                print(f"  Validation score: {metrics['validation_score']:.3f}")

        except Exception as e:
            if verbose:
                print(f"✗ Analysis failed: {e}")

            metrics = {
                "analysis_type": ground_truth.analysis_type,
                "success": False,
                "error": str(e),
            }

        return metrics

    def test_dataset(self, dataset_dir, analysis_type="sessile"):
        """
        Test analyzer on complete dataset

        Parameters:
        -----------
        dataset_dir : str or Path
            Directory containing test images and metadata.json
        analysis_type : str
            'sessile', 'pendant', 'capillary', or 'captive'
        """
        dataset_dir = Path(dataset_dir)

        # Load metadata
        with open(dataset_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        print(f"\nTesting {analysis_type} dataset: {len(metadata)} images")
        print("=" * 60)

        results = []

        for i, item in enumerate(metadata):
            print(f"\nImage {i+1}/{len(metadata)}: {item['filename']}")

            # Load image
            import cv2

            image = cv2.imread(
                str(dataset_dir / item["filename"]), cv2.IMREAD_GRAYSCALE
            )

            # Reconstruct ground truth
            from synthetic_test_image_generator import GroundTruth

            gt_dict = item["ground_truth"]
            ground_truth = GroundTruth(**gt_dict)

            # Test
            metrics = self.test_single_image(image, ground_truth, verbose=True)
            metrics["filename"] = item["filename"]
            metrics["parameters"] = item["parameters"]

            results.append(metrics)

        self.results.extend(results)

        # Generate report
        self._generate_report(results, analysis_type)

        return results

    def _generate_report(self, results, analysis_type):
        """Generate validation report"""
        df = pd.DataFrame(results)

        # Filter successful tests
        df_success = df[df["success"] == True]

        print("\n" + "=" * 60)
        print(f"VALIDATION REPORT - {analysis_type.upper()}")
        print("=" * 60)

        print(
            f"\nSuccess rate: {len(df_success)}/{len(df)} ({100*len(df_success)/len(df):.1f}%)"
        )

        if len(df_success) == 0:
            print("No successful analyses to report")
            return

        # Report metrics by analysis type
        if analysis_type == "sessile":
            self._report_sessile_metrics(df_success)
        elif analysis_type == "pendant":
            self._report_pendant_metrics(df_success)
        elif analysis_type == "capillary":
            self._report_capillary_metrics(df_success)
        elif analysis_type == "captive":
            self._report_captive_metrics(df_success)

        print("\nValidation scores:")
        print(f"  Mean: {df_success['validation_score'].mean():.3f}")
        print(f"  Std:  {df_success['validation_score'].std():.3f}")
        print(f"  Min:  {df_success['validation_score'].min():.3f}")
        print(f"  Max:  {df_success['validation_score'].max():.3f}")

    def _report_sessile_metrics(self, df):
        """Report sessile drop metrics"""
        print("\nContact Angle:")
        print(f"  MAE:  {df['contact_angle_abs_error'].mean():.2f}°")
        print(f"  RMSE: {np.sqrt((df['contact_angle_abs_error']**2).mean()):.2f}°")
        print(f"  Max error: {df['contact_angle_abs_error'].max():.2f}°")
        print(f"  Relative error: {df['contact_angle_rel_error'].mean():.1f}%")

        if "volume_abs_error" in df.columns:
            print("\nVolume:")
            print(f"  MAE:  {df['volume_abs_error'].mean():.3f} μL")
            print(f"  Relative error: {df['volume_rel_error'].mean():.1f}%")

        if "contact_diameter_abs_error" in df.columns:
            print("\nContact Diameter:")
            print(f"  MAE:  {df['contact_diameter_abs_error'].mean():.3f} mm")
            print(f"  Relative error: {df['contact_diameter_rel_error'].mean():.1f}%")

        # Check if error depends on contact angle
        print("\nError vs Contact Angle:")
        for theta_range in [(0, 45), (45, 90), (90, 135), (135, 180)]:
            mask = (df["contact_angle_true"] >= theta_range[0]) & (
                df["contact_angle_true"] < theta_range[1]
            )
            if mask.sum() > 0:
                mae = df[mask]["contact_angle_abs_error"].mean()
                print(f"  {theta_range[0]}-{theta_range[1]}°: MAE = {mae:.2f}°")

    def _report_pendant_metrics(self, df):
        """Report pendant drop metrics"""
        print("\nSurface Tension:")
        print(f"  MAE:  {df['surface_tension_abs_error'].mean():.2f} mN/m")
        print(f"  Relative error: {df['surface_tension_rel_error'].mean():.1f}%")

        if "bond_number_abs_error" in df.columns:
            print("\nBond Number:")
            print(f"  MAE:  {df['bond_number_abs_error'].mean():.4f}")
            print(f"  Relative error: {df['bond_number_rel_error'].mean():.1f}%")

    def _report_capillary_metrics(self, df):
        """Report capillary rise metrics"""
        print("\nRise Height:")
        print(f"  MAE:  {df['rise_height_abs_error'].mean():.2f} mm")
        print(f"  Relative error: {df['rise_height_rel_error'].mean():.1f}%")

        print("\nContact Angle:")
        print(f"  MAE:  {df['contact_angle_abs_error'].mean():.2f}°")

    def _report_captive_metrics(self, df):
        """Report captive bubble metrics"""
        print("\nContact Angle:")
        print(f"  MAE:  {df['contact_angle_abs_error'].mean():.2f}°")
        print(f"  Relative error: {df['contact_angle_rel_error'].mean():.1f}%")

    def plot_results(self, analysis_type="sessile", save_path=None):
        """Plot validation results"""
        df = pd.DataFrame(
            [r for r in self.results if r["analysis_type"] == analysis_type]
        )
        df_success = df[df["success"] == True]

        if len(df_success) == 0:
            print("No successful results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        if analysis_type == "sessile":
            # Contact angle: measured vs true
            axes[0, 0].scatter(
                df_success["contact_angle_true"],
                df_success["contact_angle_measured"],
                alpha=0.6,
            )
            axes[0, 0].plot([0, 180], [0, 180], "r--", label="Perfect")
            axes[0, 0].set_xlabel("True Contact Angle (°)")
            axes[0, 0].set_ylabel("Measured Contact Angle (°)")
            axes[0, 0].set_title("Contact Angle: Measured vs True")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Error vs true value
            axes[0, 1].scatter(
                df_success["contact_angle_true"],
                df_success["contact_angle_abs_error"],
                alpha=0.6,
            )
            axes[0, 1].axhline(y=2, color="orange", linestyle="--", label="±2° target")
            axes[0, 1].set_xlabel("True Contact Angle (°)")
            axes[0, 1].set_ylabel("Absolute Error (°)")
            axes[0, 1].set_title("Contact Angle Error")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Volume comparison
            if "volume_measured" in df_success.columns:
                axes[1, 0].scatter(
                    df_success["volume_true"], df_success["volume_measured"], alpha=0.6
                )
                max_vol = max(
                    df_success["volume_true"].max(), df_success["volume_measured"].max()
                )
                axes[1, 0].plot([0, max_vol], [0, max_vol], "r--", label="Perfect")
                axes[1, 0].set_xlabel("True Volume (μL)")
                axes[1, 0].set_ylabel("Measured Volume (μL)")
                axes[1, 0].set_title("Volume: Measured vs True")
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Validation score histogram
            axes[1, 1].hist(
                df_success["validation_score"], bins=20, alpha=0.7, edgecolor="black"
            )
            axes[1, 1].axvline(
                x=0.85, color="green", linestyle="--", label="Good (>0.85)"
            )
            axes[1, 1].axvline(
                x=0.7, color="orange", linestyle="--", label="Acceptable (>0.7)"
            )
            axes[1, 1].set_xlabel("Validation Score")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Validation Score Distribution")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        elif analysis_type == "pendant":
            # Surface tension
            axes[0, 0].scatter(
                df_success["surface_tension_true"],
                df_success["surface_tension_measured"],
                alpha=0.6,
            )
            max_gamma = max(
                df_success["surface_tension_true"].max(),
                df_success["surface_tension_measured"].max(),
            )
            axes[0, 0].plot([0, max_gamma], [0, max_gamma], "r--", label="Perfect")
            axes[0, 0].set_xlabel("True Surface Tension (mN/m)")
            axes[0, 0].set_ylabel("Measured Surface Tension (mN/m)")
            axes[0, 0].set_title("Surface Tension: Measured vs True")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Error
            axes[0, 1].scatter(
                df_success["surface_tension_true"],
                df_success["surface_tension_abs_error"],
                alpha=0.6,
            )
            axes[0, 1].axhline(
                y=0.5, color="orange", linestyle="--", label="±0.5 mN/m target"
            )
            axes[0, 1].set_xlabel("True Surface Tension (mN/m)")
            axes[0, 1].set_ylabel("Absolute Error (mN/m)")
            axes[0, 1].set_title("Surface Tension Error")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Validation scores
            axes[1, 0].hist(
                df_success["validation_score"], bins=20, alpha=0.7, edgecolor="black"
            )
            axes[1, 0].set_xlabel("Validation Score")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("Validation Score Distribution")
            axes[1, 0].grid(True, alpha=0.3)

            # Error distribution
            axes[1, 1].hist(
                df_success["surface_tension_rel_error"],
                bins=20,
                alpha=0.7,
                edgecolor="black",
            )
            axes[1, 1].set_xlabel("Relative Error (%)")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Relative Error Distribution")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.savefig(
                f"{analysis_type}_validation_results.png", dpi=150, bbox_inches="tight"
            )
            print(f"Plot saved to '{analysis_type}_validation_results.png'")

        plt.close()

    def export_results(self, filename="validation_results.csv"):
        """Export all results to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


def demo_validation():
    """
    Demonstration of validation framework
    This is a template - adapt to your actual ASDA implementation
    """
    print("=" * 60)
    print("ASDA VALIDATION FRAMEWORK DEMO")
    print("=" * 60)

    print(
        """
This validation framework will:
1. Load synthetic test images with known ground truth
2. Run your ASDA analyzer on each image
3. Compare results to ground truth
4. Calculate error metrics (MAE, RMSE, relative error)
5. Generate validation reports and plots
6. Export results to CSV

To use with your implementation:
    
from asda_validation_framework import ASDATester
from your_asda_library import SessileDropAnalyzer

# Create tester
tester = ASDATester(SessileDropAnalyzer, calibration=20.0)

# Test on dataset
results = tester.test_dataset('test_dataset/sessile', analysis_type='sessile')

# Plot results
tester.plot_results(analysis_type='sessile')

# Export
tester.export_results('validation_results.csv')

Acceptance Criteria:
--------------------
Sessile Drop:
  - Contact angle: MAE < 2°, relative error < 5%
  - Volume: relative error < 5%
  - Validation score: > 0.85

Pendant Drop:
  - Surface tension: MAE < 0.5 mN/m, relative error < 2%
  - Bond number: MAE < 0.05
  - Validation score: > 0.85

Capillary Rise:
  - Rise height: MAE < 0.5 mm, relative error < 5%
  - Contact angle: MAE < 5°
  
Captive Bubble:
  - Contact angle: MAE < 2°, relative error < 5%
    """
    )


if __name__ == "__main__":
    demo_validation()
