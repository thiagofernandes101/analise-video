"""
Tkinter-based GUI Summary Window - Displays video analysis summary in a graphical window.

Provides a scrollable window with all statistics that stays open until manually closed.
Works with WSLg on Windows 11.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional

from models.video_statistics import VideoStatistics


class SummaryGUIWindow:
    """
    GUI window for displaying video analysis summary.
    
    Uses Tkinter to create a resizable, scrollable window with:
    - Title and statistics
    - Scrollable text area with all details
    - Close button
    """
    
    def __init__(self, title: str = "Video Analysis Summary"):
        """
        Initialize the GUI window.
        
        Args:
            title: Window title
        """
        self.title = title
        self.root: Optional[tk.Tk] = None
    
    def show(self, statistics: VideoStatistics) -> None:
        """
        Display the summary window and wait for user to close it.
        
        Args:
            statistics: Video statistics to display
        """
        # Create main window
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("900x700")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title label
        title_label = ttk.Label(
            main_frame,
            text="📊 VIDEO ANALYSIS SUMMARY",
            font=('Arial', 16, 'bold'),
            foreground='#0066cc'
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Overview", padding="10")
        stats_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Overview statistics
        stats_text = f"""
Total Frames Analyzed: {statistics.total_frames}
Persons Detected: {statistics.get_person_count()}
Anomalies Detected: {statistics.get_anomaly_count()}
        """.strip()
        
        stats_label = ttk.Label(stats_frame, text=stats_text, font=('Courier', 10))
        stats_label.grid(row=0, column=0, sticky=tk.W)
        
        # Scrollable text area for detailed summary
        text_frame = ttk.LabelFrame(main_frame, text="Detailed Analysis", padding="10")
        text_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(2, weight=1)
        
        # Create scrolled text widget
        text_area = scrolledtext.ScrolledText(
            text_frame,
            width=100,
            height=30,
            font=('Courier', 9),
            wrap=tk.WORD,
            bg='#f5f5f5'
        )
        text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        # Build detailed summary text
        summary_text = self._build_summary_text(statistics)
        text_area.insert('1.0', summary_text)
        text_area.config(state='disabled')  # Make read-only
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        # Close button
        close_btn = ttk.Button(
            button_frame,
            text="Close",
            command=self.root.destroy,
            width=15
        )
        close_btn.pack()
        
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
        # Start the GUI event loop
        self.root.mainloop()
    
    def _build_summary_text(self, stats: VideoStatistics) -> str:
        """
        Build detailed summary text.
        
        Args:
            stats: Video statistics
            
        Returns:
            Formatted summary text
        """
        lines = []
        lines.append("=" * 90)
        lines.append("TOP ACTIVITIES")
        lines.append("=" * 90)
        
        for activity, count in stats.get_top_activities():
            bar_length = int((count / max(c for _, c in stats.get_top_activities())) * 40)
            bar = "█" * bar_length
            lines.append(f"  {activity:.<50} {count:>6}  {bar}")
        
        lines.append("")
        lines.append("=" * 90)
        lines.append("TOP EMOTIONS")
        lines.append("=" * 90)
        
        for emotion, count in stats.get_top_emotions():
            bar_length = int((count / max(c for _, c in stats.get_top_emotions())) * 40)
            bar = "█" * bar_length
            lines.append(f"  {emotion:.<50} {count:>6}  {bar}")
        
        lines.append("")
        lines.append("=" * 90)
        lines.append("PER-PERSON ANALYSIS")
        lines.append("=" * 90)
        
        for person in stats.get_sorted_persons()[:15]:  # Limit to first 15
            lines.append("")
            lines.append(f"PERSON ID #{person.track_id} ({person.frame_count} frames)")
            lines.append(f"  Emotions:   {person.get_emotions_display()}")
            lines.append(f"  Activities: {person.get_activities_display()}")
            
            if person.anomalies:
                lines.append(f"  Anomalies:  {len(person.anomalies)}")
                for anomaly in person.anomalies[:5]:  # First 5 anomalies per person
                    lines.append(f"    • Frame {anomaly.frame_number}: {anomaly.explanation}")
                if len(person.anomalies) > 5:
                    lines.append(f"    ... and {len(person.anomalies) - 5} more")
        
        if stats.get_person_count() > 15:
            lines.append(f"\n... and {stats.get_person_count() - 15} more persons")
        
        if stats.all_anomalies:
            lines.append("")
            lines.append("=" * 90)
            lines.append(f"ALL ANOMALIES ({len(stats.all_anomalies)} total)")
            lines.append("=" * 90)
            
            for anomaly in stats.all_anomalies[:20]:  # First 20 anomalies
                lines.append(
                    f"  Frame {anomaly.frame_number:>5}, "
                    f"Person #{anomaly.track_id:>2}: {anomaly.explanation}"
                )
            
            if len(stats.all_anomalies) > 20:
                lines.append(f"\n... and {len(stats.all_anomalies) - 20} more anomalies")
        
        lines.append("")
        lines.append("=" * 90)
        lines.append("END OF SUMMARY")
        lines.append("=" * 90)
        
        return "\n".join(lines)
