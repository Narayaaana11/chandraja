"""Chart generation for visualization of results."""

import logging
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generate charts for visualizing evaluation results."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize chart generator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        # Set matplotlib to use non-interactive backend
        plt.switch_backend('Agg')
    
    def bar_chart(self, question_scores: Dict[str, Dict[str, int]],
                 output_path: str) -> str:
        """
        Generate bar chart showing marks per question.
        
        Args:
            question_scores: Dictionary with structure:
            {
                "Q1": {"marks_awarded": 8, "max_marks": 10},
                "Q2": {"marks_awarded": 5, "max_marks": 10},
                ...
            }
            output_path: Path to save PNG file
        
        Returns:
            Path to saved chart file
        """
        try:
            # Extract data
            questions = list(question_scores.keys())
            marks_awarded = [question_scores[q]["marks_awarded"] for q in questions]
            max_marks = [question_scores[q]["max_marks"] for q in questions]
            
            # Calculate colors based on percentage
            colors = []
            for m, max_m in zip(marks_awarded, max_marks):
                percentage = (m / max_m * 100) if max_m > 0 else 0
                if percentage >= 70:
                    colors.append('#4CAF50')  # Green
                elif percentage >= 40:
                    colors.append('#FFC107')  # Amber
                else:
                    colors.append('#F44336')  # Red
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = range(len(questions))
            width = 0.35
            
            # Plot bars
            bars1 = ax.bar([i - width/2 for i in x], marks_awarded, width,
                          label='Marks Awarded', color=colors, alpha=0.8)
            bars2 = ax.bar([i + width/2 for i in x], max_marks, width,
                          label='Max Marks', color='lightgray', alpha=0.6)
            
            # Customize chart
            ax.set_xlabel('Questions', fontsize=12, fontweight='bold')
            ax.set_ylabel('Marks', fontsize=12, fontweight='bold')
            ax.set_title('Marks per Question', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(questions)
            ax.legend(loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Create output directory if needed
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Bar chart saved to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating bar chart: {e}")
            raise
    
    def line_chart(self, submissions: List[Dict[str, Any]], output_path: str) -> str:
        """
        Generate line chart showing performance trend over submissions.
        
        Args:
            submissions: List of submission dictionaries with structure:
            {
                "timestamp": "2024-01-01T10:00:00",
                "total_score": 78,
                "out_of": 100,
                ...
            }
            output_path: Path to save PNG file
        
        Returns:
            Path to saved chart file
        """
        try:
            if not submissions:
                logger.warning("No submissions to plot")
                return ""
            
            # Extract data
            percentages = [
                (s["total_score"] / s["out_of"] * 100) if s["out_of"] > 0 else 0
                for s in submissions
            ]
            timestamps = [s.get("timestamp", f"Submission {i+1}")
                         for i, s in enumerate(submissions)]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = range(len(submissions))
            ax.plot(x, percentages, marker='o', linestyle='-', linewidth=2,
                   markersize=8, color='#2196F3', label='Score %')
            
            # Add threshold lines
            ax.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Pass (70%)')
            ax.axhline(y=90, color='gold', linestyle='--', alpha=0.5, label='Excellent (90%)')
            
            # Customize chart
            ax.set_xlabel('Submission', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage Score (%)', fontsize=12, fontweight='bold')
            ax.set_title('Performance Trend Across Submissions', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f"S{i+1}" for i in x])
            ax.set_ylim(0, 105)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on points
            for i, pct in enumerate(percentages):
                ax.text(i, pct + 2, f'{pct:.0f}%', ha='center', fontsize=9)
            
            plt.tight_layout()
            
            # Create output directory if needed
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Line chart saved to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error generating line chart: {e}")
            raise
    
    def pie_chart(self, total_score: int, max_score: int, output_path: str) -> str:
        """Generate pie chart showing scored vs missed marks."""
        try:
            total = float(total_score)
            maximum = float(max_score)
            total = max(total, 0.0)
            maximum = max(maximum, 0.0)
            if total > maximum:
                total = maximum

            percentage = (total / maximum * 100) if maximum > 0 else 0.0
            missed = max(maximum - total, 0.0)

            if total == 0 and missed == 0:
                total = 1.0

            fig, ax = plt.subplots(figsize=(8, 8))
            sizes = [total, missed]
            colors = ['#4CAF50', '#E0E0E0']
            labels = [f'Scored\n{total:.2f}', f'Missed\n{missed:.2f}']

            ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 11, 'weight': 'bold'}
            )

            centre_circle = plt.Circle((0, 0), 0.70, fc='white')
            ax.add_artist(centre_circle)

            ax.text(
                0,
                0,
                f'{total:.2f}/{maximum:.2f}\n({percentage:.0f}%)',
                ha='center',
                va='center',
                fontsize=16,
                fontweight='bold'
            )
            ax.set_title(f'Score: {total:.2f}/{maximum:.2f}', fontsize=14, fontweight='bold')

            plt.tight_layout()
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Pie chart saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating pie chart: {e}")
            raise
    
    def generate_all(self, result: Dict[str, Any], output_dir: str = "static/charts") -> Dict[str, str]:
        """
        Generate all charts for a submission result.
        
        Args:
            result: Evaluation result dictionary containing:
            {
                "submission_id": "...",
                "question_results": {"Q1": {...}, ...},
                "total": 78,
                "out_of": 100,
                ...
            }
            output_dir: Directory to save charts
        
        Returns:
            Dictionary with paths to generated charts:
            {
                "bar_chart": "/static/charts/...",
                "pie_chart": "/static/charts/...",
                "line_chart": "/static/charts/..."
            }
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            submission_id = result.get("submission_id", "unknown")
            
            # Generate bar chart
            bar_path = output_dir / f"{submission_id}_bar.png"
            question_scores = result.get("question_results") or result.get("questions", {})
            self.bar_chart(question_scores, str(bar_path))
            
            # Generate pie chart
            pie_path = output_dir / f"{submission_id}_pie.png"
            self.pie_chart(result["total"], result["out_of"], str(pie_path))
            
            # Generate line chart (need historical data - placeholder)
            line_path = output_dir / f"{submission_id}_line.png"
            # This would need historical submissions data
            
            logger.info(f"All charts generated for submission {submission_id}")
            
            return {
                "bar_chart": str(bar_path),
                "pie_chart": str(pie_path),
                "line_chart": str(line_path)
            }
        
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            raise
