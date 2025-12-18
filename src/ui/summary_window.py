"""
SummaryWindow - Displays video analysis summary with detailed statistics.

Implements dual-view interface:
- Overview: Quick stats with bar charts
- Detailed: Per-person analysis with anomaly explanations

Single Responsibility: Summary UI rendering and user interaction.
"""
import cv2 as cv
import numpy as np
from typing import Optional, Tuple, List

from models.video_statistics import VideoStatistics
from config import Config


class SummaryWindow:
    """
    Renders and manages video analysis summary window.
    
    Provides dual-view interface with overview and detailed analysis.
    Handles button clicks and window events.
    """
    
    def __init__(self, config: Config = Config()):
        """
        Initialize summary window.
        
        Args:
            config: Configuration object
        """
        self._config = config
        self._current_view = "overview"  # or "detailed"
        self._scroll_offset = 0
        self._mouse_callback_registered = False
        
        # Button coordinates (set during rendering)
        self._buttons: List[dict] = []
    
    def show_and_wait(self, statistics: VideoStatistics) -> None:
        """
        Show summary window and wait for user to close it.
        
        Args:
            statistics: Video statistics to display
        """
        import os
        import traceback
        
        # Check if running in Docker - multiple detection methods
        in_docker = (
            os.path.exists('/.dockerenv') or 
            os.path.exists('/run/.containerenv') or
            os.environ.get('DOCKER_CONTAINER') == 'true'
        )
        
        display_available = os.environ.get('DISPLAY')
        
        print(f"\n[DEBUG] Environment check:")
        print(f"  - DISPLAY: {display_available}")
        print(f"  - In Docker: {in_docker}")
        print(f"  - /.dockerenv exists: {os.path.exists('/.dockerenv')}")
        
        # If in Docker, save to files instead of showing GUI
        if in_docker:
            print("\n[INFO] Running in Docker - exporting summary to files...")
            try:
                from ui.file_summary_exporter import FileSummaryExporter
                
                # Create output directory if it doesn't exist
                output_dir = "/app/output"
                os.makedirs(output_dir, exist_ok=True)
                
                # Export to both text and JSON
                text_path = os.path.join(output_dir, "summary.txt")
                json_path = os.path.join(output_dir, "summary.json")
                
                FileSummaryExporter.export_to_text(statistics, text_path)
                FileSummaryExporter.export_to_json(statistics, json_path)
                
                print(f"\n📄 Summary files created:")
                print(f"   - Text: {text_path}")
                print(f"   - JSON: {json_path}")
                print("\n💡 Tip: You can access these files from your host machine")
                print("    at ./output/summary.txt and ./output/summary.json")
                print("\n" + "="*80)
                
                # Also print a quick summary to console
                print("\n📊 QUICK SUMMARY:")
                print(f"   Total Frames:       {statistics.total_frames}")
                print(f"   Persons Detected:   {statistics.get_person_count()}")
                print(f"   Anomalies Detected: {statistics.get_anomaly_count()}")
                
                # Top activities
                print("\n🏃 Top 5 Activities:")
                for activity, count in statistics.get_top_activities()[:5]:
                    print(f"   • {activity}: {count}")
                
                # Top emotions
                print("\n😊 Top 5 Emotions:")
                for emotion, count in statistics.get_top_emotions()[:5]:
                    print(f"   • {emotion}: {count}")
                
                print("\n" + "="*80)
                print("✅ Analysis complete! Check the files above for full details.")
                print("="*80 + "\n")
                
                return
            except Exception as e:
                print(f"\n⚠️  File export failed: {e}")
                print("Falling back to terminal UI...")
                traceback.print_exc()
                # Fall through to terminal UI
        
        # Fallback to terminal UI  
        if in_docker or not display_available:
            print("[INFO] Using terminal UI")
            try:
                self._print_text_summary(statistics)
            except Exception as e:
                print(f"\n⚠️  ERROR in terminal UI rendering:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {e}")
                traceback.print_exc()
                print("\nFalling back to basic text summary...")
                self._print_basic_summary(statistics)
            return
        
        # If not in Docker and no DISPLAY, use terminal UI
        if not display_available:
            print("[INFO] No DISPLAY - using terminal UI")
            try:
                self._print_text_summary(statistics)
            except Exception as e:
                print(f"\nERROR in terminal UI rendering:")
                traceback.print_exc()
                self._print_basic_summary(statistics)
            return
        
        window_name = "Resumo da Análise"
        
        try:
            # Render first frame
            if self._current_view == "overview":
                frame = self._render_overview(statistics)
            else:
                frame = self._render_detailed(statistics)
            
            # Try to create window
            cv.imshow(window_name, frame)
            cv.waitKey(1)  # Give window time to initialize
            
            # Verify window was actually created
            try:
                window_property = cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE)
                if window_property < 0:
                    raise Exception("Window was not created (getWindowProperty returned negative)")
            except:
                raise Exception("OpenCV window creation failed - GUI backend not available")
            
            # Now it's safe to set mouse callback
            cv.setMouseCallback(window_name, self._mouse_callback)
            self._mouse_callback_registered = True
            
            # Main loop
            while True:
                # Render current view
                if self._current_view == "overview":
                    frame = self._render_overview(statistics)
                else:
                    frame = self._render_detailed(statistics)
                
                # Display
                cv.imshow(window_name, frame)
                
                # Handle keyboard input
                key = cv.waitKey(50) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
            
            cv.destroyWindow(window_name)
        except Exception as e:
            print(f"Error displaying summary window: {e}")
            print("Displaying text summary instead...")
            # Fallback to text summary
            self._print_text_summary(statistics)
    
    def _mouse_callback(self, event, x, y, flags, param) -> None:
        """Handle mouse events for button clicks."""
        if event == cv.EVENT_LBUTTONDOWN:
            self._handle_click(x, y)
    
    def _handle_click(self, x: int, y: int) -> None:
        """
        Handle mouse click on buttons.
        
        Args:
            x: Click x coordinate
            y: Click y coordinate
        """
        for button in self._buttons:
            bx, by, bw, bh = button['rect']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                action = button['action']
                
                if action == "close":
                    cv.destroyAllWindows()
                elif action == "detailed":
                    self._current_view = "detailed"
                    self._scroll_offset = 0
                elif action == "overview":
                    self._current_view = "overview"
                    self._scroll_offset = 0
                
                break
    
    def _render_overview(self, stats: VideoStatistics) -> np.ndarray:
        """
        Render overview screen with summary statistics.
        
        Args:
            stats: Video statistics
            
        Returns:
            Rendered frame
        """
        cfg = self._config.summary
        width = cfg.SUMMARY_WINDOW_WIDTH
        height = cfg.SUMMARY_WINDOW_HEIGHT
        
        # Create white canvas
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        self._buttons = []
        
        y_pos = 20
        x_margin = 30
        
        # Title
        title = "Resumo da Análise de Vídeo"
        cv.putText(frame, title, (x_margin, y_pos), cv.FONT_HERSHEY_SIMPLEX,
                   cfg.SUMMARY_FONT_SCALE_LARGE, (0, 0, 0), 2)
        y_pos += 50
        
        # Divider line
        cv.line(frame, (x_margin, y_pos), (width - x_margin, y_pos), (200, 200, 200), 2)
        y_pos += 30
        
        # Statistics
        stats_text = [
            f"Total de Frames: {stats.total_frames}",
            f"Pessoas Detectadas: {stats.get_person_count()}",
            f"Anomalias Detectadas: {stats.get_anomaly_count()}"
        ]
        
        for text in stats_text:
            cv.putText(frame, text, (x_margin, y_pos), cv.FONT_HERSHEY_SIMPLEX,
                       cfg.SUMMARY_FONT_SCALE, (0, 0, 0), 1)
            y_pos += 35
        
        y_pos += 20
        
        # Activities section
        y_pos = self._render_distribution_section(
            frame, "ATIVIDADES PRINCIPAIS:", stats.get_top_activities(),
            y_pos, x_margin, cfg
        )
        
        y_pos += 30
        
        # Emotions section
        y_pos = self._render_distribution_section(
            frame, "EMOÇÕES PRINCIPAIS:", stats.get_top_emotions(),
            y_pos, x_margin, cfg
        )
        
        # Buttons
        y_pos = height - 80
        
        # "Ver Análise Detalhada" button
        button_width = cfg.SUMMARY_BUTTON_WIDTH
        button_height = cfg.SUMMARY_BUTTON_HEIGHT
        button_x = x_margin
        
        cv.rectangle(frame, (button_x, y_pos),
                     (button_x + button_width, y_pos + button_height),
                     (100, 100, 255), -1)
        cv.putText(frame, "Ver Analise Detalhada", (button_x + 10, y_pos + 27),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self._buttons.append({
            'rect': (button_x, y_pos, button_width, button_height),
            'action': 'detailed'
        })
        
        # "Fechar" button
        button_x = width - x_margin - button_width
        cv.rectangle(frame, (button_x, y_pos),
                     (button_x + button_width, y_pos + button_height),
                     (50, 50, 50), -1)
        cv.putText(frame, "Fechar", (button_x + 60, y_pos + 27),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self._buttons.append({
            'rect': (button_x, y_pos, button_width, button_height),
            'action': 'close'
        })
        
        return frame
    
    def _render_distribution_section(
        self,
        frame: np.ndarray,
        title: str,
        items: List[Tuple[str, int]],
        y_pos: int,
        x_margin: int,
        cfg
    ) -> int:
        """
        Render a distribution section with bars.
        
        Args:
            frame: Frame to render on
            title: Section title
            items: List of (label, count) tuples
            y_pos: Starting y position
            x_margin: Left margin
            cfg: Summary config
            
        Returns:
            New y position after rendering
        """
        # Section title
        cv.putText(frame, title, (x_margin, y_pos), cv.FONT_HERSHEY_SIMPLEX,
                   cfg.SUMMARY_FONT_SCALE, (0, 0, 0), 2)
        y_pos += 30
        
        if not items:
            cv.putText(frame, "  Nenhum dado disponivel", (x_margin, y_pos),
                       cv.FONT_HERSHEY_SIMPLEX, cfg.SUMMARY_FONT_SCALE,
                       (100, 100, 100), 1)
            return y_pos + 30
        
        # Find max count for scaling
        max_count = max(count for _, count in items) if items else 1
        
        for label, count in items:
            # Bar
            bar_width = int((count / max_count) * cfg.SUMMARY_BAR_MAX_WIDTH)
            cv.rectangle(frame, (x_margin + 10, y_pos - 15),
                         (x_margin + 10 + bar_width, y_pos + 5),
                         (100, 150, 255), -1)
            
            # Label and count
            text = f"{label} ({count})"
            cv.putText(frame, text, (x_margin + 20 + bar_width, y_pos),
                       cv.FONT_HERSHEY_SIMPLEX, cfg.SUMMARY_FONT_SCALE - 0.1,
                       (0, 0, 0), 1)
            
            y_pos += 30
        
        return y_pos
    
    def _render_detailed(self, stats: VideoStatistics) -> np.ndarray:
        """
        Render detailed analysis view.
        
        Args:
            stats: Video statistics
            
        Returns:
            Rendered frame
        """
        cfg = self._config.summary
        width = cfg.DETAIL_WINDOW_WIDTH
        height = cfg.DETAIL_WINDOW_HEIGHT
        
        # Create white canvas (larger virtual canvas for scrolling)
        virtual_height = height * 3  # Enough for scrolling
        virtual_frame = np.ones((virtual_height, width, 3), dtype=np.uint8) * 255
        self._buttons = []
        
        y_pos = 20
        x_margin = 30
        indent = cfg.DETAIL_INDENT
        
        # Title
        title = "Analise Detalhada"
        cv.putText(virtual_frame, title, (x_margin, y_pos), cv.FONT_HERSHEY_SIMPLEX,
                   cfg.SUMMARY_FONT_SCALE_LARGE, (0, 0, 0), 2)
        y_pos += 50
        
        # Divider
        cv.line(virtual_frame, (x_margin, y_pos), (width - x_margin, y_pos),
                (200, 200, 200), 2)
        y_pos += 30
        
        # Per-person statistics
        persons = stats.get_sorted_persons()
        
        for person_stats in persons:
            # Person header
            header = f"PESSOA ID #{person_stats.track_id} ({person_stats.frame_count} frames)"
            cv.putText(virtual_frame, header, (x_margin, y_pos),
                       cv.FONT_HERSHEY_SIMPLEX, cfg.DETAIL_FONT_SCALE, (0, 50, 150), 2)
            y_pos += 25
            
            # Emotions
            emotions_text = f"Emocoes: {person_stats.get_emotions_display()}"
            cv.putText(virtual_frame, emotions_text, (x_margin + indent, y_pos),
                       cv.FONT_HERSHEY_SIMPLEX, cfg.DETAIL_FONT_SCALE, (0, 0, 0), 1)
            y_pos += 22
            
            # Activities
            activities_text = f"Atividades: {person_stats.get_activities_display()}"
            cv.putText(virtual_frame, activities_text, (x_margin + indent, y_pos),
                       cv.FONT_HERSHEY_SIMPLEX, cfg.DETAIL_FONT_SCALE, (0, 0, 0), 1)
            y_pos += 22
            
            # Anomalies
            anomaly_count = len(person_stats.anomalies)
            anomaly_text = f"Anomalias: {anomaly_count}"
            cv.putText(virtual_frame, anomaly_text, (x_margin + indent, y_pos),
                       cv.FONT_HERSHEY_SIMPLEX, cfg.DETAIL_FONT_SCALE,
                       (200, 0, 0) if anomaly_count > 0 else (0, 0, 0), 1)
            y_pos += 22
            
            # List anomalies
            for anomaly in person_stats.anomalies[:3]:  # Limit to first 3
                bullet = f"  Frame {anomaly.frame_number}: {anomaly.explanation}"
                cv.putText(virtual_frame, bullet, (x_margin + indent * 2, y_pos),
                           cv.FONT_HERSHEY_SIMPLEX, cfg.DETAIL_FONT_SCALE - 0.05,
                           (150, 0, 0), 1)
                y_pos += 20
            
            # Movement segments timeline
            if person_stats.movement_segments:
                cv.putText(virtual_frame, "Timeline:", (x_margin + indent, y_pos),
                           cv.FONT_HERSHEY_SIMPLEX, cfg.DETAIL_FONT_SCALE, (0, 100, 0), 1)
                y_pos += 22
                
                for seg_line in person_stats.get_segments_display()[:5]:  # Limit to first 5
                    cv.putText(virtual_frame, f"  {seg_line}", (x_margin + indent * 2, y_pos),
                               cv.FONT_HERSHEY_SIMPLEX, cfg.DETAIL_FONT_SCALE - 0.05,
                               (0, 80, 0) if "⚠️" not in seg_line else (200, 100, 0), 1)
                    y_pos += 20
            
            y_pos += 15
        
        #Anomalies summary section
        if stats.all_anomalies:
            y_pos += 10
            cv.putText(virtual_frame, f"ANOMALIAS ({len(stats.all_anomalies)} total)",
                       (x_margin, y_pos), cv.FONT_HERSHEY_SIMPLEX,
                       cfg.DETAIL_FONT_SCALE, (200, 0, 0), 2)
            y_pos += 25
            
            for anomaly in stats.all_anomalies[:20]:  # Limit to first 20
                short_desc = anomaly.get_short_description()
                cv.putText(virtual_frame, f"  {short_desc}",
                           (x_margin + indent, y_pos),
                           cv.FONT_HERSHEY_SIMPLEX, cfg.DETAIL_FONT_SCALE - 0.05,
                           (100, 0, 0), 1)
                y_pos += 20
        
        # Extract visible portion
        visible_y_start = min(self._scroll_offset, virtual_height - height)
        visible_y_end = visible_y_start + height
        frame = virtual_frame[visible_y_start:visible_y_end, :].copy()
        
        # Buttons (always at bottom of visible area)
        button_y = height - 60
        button_width = 150
        button_height = 40
        
        # "Voltar" button
        button_x = x_margin
        cv.rectangle(frame, (button_x, button_y),
                     (button_x + button_width, button_y + button_height),
                     (100, 100, 100), -1)
        cv.putText(frame, "<- Voltar", (button_x + 30, button_y + 27),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self._buttons.append({
            'rect': (button_x, button_y, button_width, button_height),
            'action': 'overview'
        })
        
        # "Fechar" button
        button_x = width - x_margin - button_width
        cv.rectangle(frame, (button_x, button_y),
                     (button_x + button_width, button_y + button_height),
                     (50, 50, 50), -1)
        cv.putText(frame, "Fechar", (button_x + 40, button_y + 27),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self._buttons.append({
            'rect': (button_x, button_y, button_width, button_height),
            'action': 'close'
        })
        
        return frame
    
    def _print_text_summary(self, stats: VideoStatistics) -> None:
        """Print beautiful text-based summary using Rich library."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich.text import Text
            from rich import box
            
            console = Console()
            
            # Main title
            console.print("\n")
            title = Text("VIDEO ANALYSIS SUMMARY", style="bold white on blue")
            console.print(Panel(title, box=box.DOUBLE, border_style="blue"))
            
            # Overview statistics
            overview_table = Table(title="📊 Overview Statistics", box=box.ROUNDED, show_header=False)
            overview_table.add_column("Metric", style="cyan", width=25)
            overview_table.add_column("Value", style="green bold", width=15)
            
            overview_table.add_row("Total Frames", str(stats.total_frames))
            overview_table.add_row("Persons Detected", str(stats.get_person_count()))
            overview_table.add_row("Anomalies Detected", 
                                  f"[red]{stats.get_anomaly_count()}[/red]" if stats.get_anomaly_count() > 0 else "0")
            
            console.print(overview_table)
            console.print()
            
            # Top Activities
            activities_table = Table(title="🏃 Top Activities", box=box.ROUNDED)
            activities_table.add_column("Activity", style="cyan", width=35)
            activities_table.add_column("Count", style="magenta", justify="right", width=10)
            activities_table.add_column("Bar", style="blue", width=30)
            
            top_activities = stats.get_top_activities()
            if top_activities:
                max_count = max(count for _, count in top_activities)
                for activity, count in top_activities:
                    bar_length = int((count / max_count) * 20)
                    bar = "█" * bar_length
                    activities_table.add_row(activity, str(count), bar)
            else:
                activities_table.add_row("No data", "0", "")
            
            console.print(activities_table)
            console.print()
            
            # Top Emotions
            emotions_table = Table(title="😊 Top Emotions", box=box.ROUNDED)
            emotions_table.add_column("Emotion", style="cyan", width=35)
            emotions_table.add_column("Count", style="magenta", justify="right", width=10)
            emotions_table.add_column("Bar", style="yellow", width=30)
            
            top_emotions = stats.get_top_emotions()
            if top_emotions:
                max_count = max(count for _, count in top_emotions)
                for emotion, count in top_emotions:
                    bar_length = int((count / max_count) * 20)
                    bar = "█" * bar_length
                    emotions_table.add_row(emotion, str(count), bar)
            else:
                emotions_table.add_row("No data", "0", "")
            
            console.print(emotions_table)
            console.print()
            
            # Per-Person Details
            if stats.get_person_count() > 0:
                person_table = Table(title="👤 Per-Person Analysis", box=box.ROUNDED)
                person_table.add_column("ID", style="cyan bold", width=5)
                person_table.add_column("Frames", style="green", width=8)
                person_table.add_column("Emotions", style="yellow", width=30)
                person_table.add_column("Activities", style="blue", width=30)
                person_table.add_column("Anomalies", style="red", width=10)
                
                for person in stats.get_sorted_persons()[:10]:  # Limit to first 10
                    emotions = person.get_emotions_display()
                    if len(emotions) > 28:
                        emotions = emotions[:25] + "..."
                    
                    activities = person.get_activities_display()
                    if len(activities) > 28:
                        activities = activities[:25] + "..."
                    
                    anomaly_count = len(person.anomalies)
                    
                    person_table.add_row(
                        f"#{person.track_id}",
                        str(person.frame_count),
                        emotions,
                        activities,
                        f"[red bold]{anomaly_count}[/]" if anomaly_count > 0 else str(anomaly_count)
                    )
                
                console.print(person_table)
                console.print()
                
                # Movement Timeline for each person
                for person in stats.get_sorted_persons()[:5]:  # Limit to first 5
                    if person.movement_segments:
                        timeline_table = Table(
                            title=f"📍 Person #{person.track_id} Movement Timeline",
                            box=box.SIMPLE
                        )
                        timeline_table.add_column("Time", style="cyan", width=15)
                        timeline_table.add_column("Activity", style="green", width=20)
                        timeline_table.add_column("Emotion", style="yellow", width=15)
                        timeline_table.add_column("Anomalies", style="red", width=30)
                        
                        for seg in person.movement_segments[:10]:
                            time_str = seg.get_duration_display()
                            anomaly_str = ", ".join(seg.anomalies) if seg.anomalies else "-"
                            timeline_table.add_row(
                                time_str,
                                seg.activity,
                                seg.dominant_emotion,
                                f"[red]{anomaly_str}[/red]" if seg.anomalies else anomaly_str
                            )
                        
                        console.print(timeline_table)
                        console.print()
            
            # Anomalies Details (if any)
            if stats.all_anomalies:
                # Split anomalies
                system_anomalies = []
                behavioral_anomalies = []
                
                for anomaly in stats.all_anomalies:
                    desc = anomaly.explanation.lower()
                    if "tracking error" in desc or "analyzing" in desc:
                        system_anomalies.append(anomaly)
                    else:
                        behavioral_anomalies.append(anomaly)
                
                # Render Behavioral Anomalies
                if behavioral_anomalies:
                    console.print(Panel(
                        f"[red bold]⚠️  {len(behavioral_anomalies)} Behavioral Anomalies Detected[/]",
                        box=box.ROUNDED,
                        border_style="red"
                    ))
                    
                    anomaly_table = Table(box=box.SIMPLE)
                    anomaly_table.add_column("Frame", style="cyan", width=8)
                    anomaly_table.add_column("Person ID", style="green", width=10)
                    anomaly_table.add_column("Description", style="yellow", width=60)
                    
                    for anomaly in behavioral_anomalies[:15]:
                        anomaly_table.add_row(
                            str(anomaly.frame_number),
                            f"#{anomaly.track_id}",
                            anomaly.explanation
                        )
                    console.print(anomaly_table)
                    console.print()

                # Render System Diagnostics (Tracking Errors)
                if system_anomalies:
                    console.print(Panel(
                        f"[orange3]🔧 System Diagnostics ({len(system_anomalies)} Tracking Errors)[/]",
                        box=box.ROUNDED,
                        border_style="orange3"
                    ))
                    
                    sys_table = Table(box=box.SIMPLE)
                    sys_table.add_column("Frame", style="dim cyan", width=8)
                    sys_table.add_column("Person ID", style="dim green", width=10)
                    sys_table.add_column("Description", style="dim white", width=60)
                    
                    for anomaly in system_anomalies[:5]:
                        sys_table.add_row(
                            str(anomaly.frame_number),
                            f"#{anomaly.track_id}",
                            anomaly.explanation
                        )
                    console.print(sys_table)
                    console.print()
            
            # Footer
            console.print(Panel(
                "[dim]Summary generated successfully. Use this data for further analysis.[/dim]",
                box=box.ROUNDED,
                border_style="dim"
            ))
            console.print()
            
            # Wait for user to press Enter before closing
            console.print("[bold cyan]Press Enter to close...[/bold cyan]")
            input()
            
        except ImportError:
            # Fallback to plain text if Rich is not available
            print("\n" + "="*50)
            print("VIDEO ANALYSIS SUMMARY")
            print("="*50)
            print(f"\nTotal Frames: {stats.total_frames}")
            print(f"Persons Detected: {stats.get_person_count()}")
            print(f"Anomalies Detected: {stats.get_anomaly_count()}")
            
            print("\n--- Top Activities ---")
            for activity, count in stats.get_top_activities():
                print(f"  {activity}: {count}")
            
            print("\n--- Top Emotions ---")
            for emotion, count in stats.get_top_emotions():
                print(f"  {emotion}: {count}")
            
            print("\n" + "="*50)


    def _print_basic_summary(self, stats: VideoStatistics) -> None:
        """Print basic text summary when Rich fails."""
        print("\n" + "="*80)
        print(" "*25 + "VIDEO ANALYSIS SUMMARY")
        print("="*80)
        print(f"\nTotal Frames: {stats.total_frames}")
        print(f"Persons Detected: {stats.get_person_count()}")
        print(f"Anomalies Detected: {stats.get_anomaly_count()}")
        
        print("\n--- Top Activities ---")
        for activity, count in stats.get_top_activities():
            print(f"  {activity}: {count}")
        
        print("\n--- Top Emotions ---")
        for emotion, count in stats.get_top_emotions():
            print(f"  {emotion}: {count}")
        
        if stats.get_person_count() > 0:
            print("\n--- Per-Person Analysis (First 5) ---")
            for i, person in enumerate(stats.get_sorted_persons()[:5], 1):
                print(f"\n  {i}. Person ID #{person.track_id} ({person.frame_count} frames)")
                print(f"     Emotions: {person.get_emotions_display()}")
                print(f"     Activities: {person.get_activities_display()}")
                if person.anomalies:
                    print(f"     Anomalies: {len(person.anomalies)}")
                    for anomaly in person.anomalies[:2]:
                        print(f"       - Frame {anomaly.frame_number}: {anomaly.explanation}")
        
        if stats.all_anomalies:
            # Split anomalies
            system_anomalies = []
            behavioral_anomalies = []
            
            for anomaly in stats.all_anomalies:
                desc = anomaly.explanation.lower()
                if "tracking error" in desc or "analyzing" in desc:
                    system_anomalies.append(anomaly)
                else:
                    behavioral_anomalies.append(anomaly)
            
            if behavioral_anomalies:
                print(f"\n--- Behavioral Anomalies ({len(behavioral_anomalies)} total) ---")
                for anomaly in behavioral_anomalies[:15]:
                    print(f"  Frame {anomaly.frame_number}, ID #{anomaly.track_id}: {anomaly.explanation}")
                    
            if system_anomalies:
                print(f"\n--- System Diagnostics / Tracking Errors ({len(system_anomalies)} total) ---")
                count = 0
                for anomaly in system_anomalies:
                    if count < 5: # Show fewer system errors
                        print(f"  Frame {anomaly.frame_number}, ID #{anomaly.track_id}: {anomaly.explanation}")
                    count += 1
                if count > 5:
                    print(f"  ... and {count - 5} more tracking errors")
        
        print("\n" + "="*80)
        print(" "*20 + "Summary generated successfully")
        print("="*80 + "\n")
        print("Press Enter to close...")
        input()
