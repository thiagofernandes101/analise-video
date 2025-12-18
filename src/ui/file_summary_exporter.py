"""
File Summary Exporter - Exports video analysis summary to a file.

This module provides functionality to export the summary to various formats
without requiring GUI or user interaction, perfect for Docker/headless environments.
"""
import json
from datetime import datetime
from typing import TextIO
from models.video_statistics import VideoStatistics


class FileSummaryExporter:
    """
    Exports video analysis summary to files.
    
    Supports both text and JSON formats.
    """
    
    @staticmethod
    def export_to_text(stats: VideoStatistics, filepath: str) -> None:
        """
        Export summary to a formatted text file.
        
        Args:
            stats: Video statistics
            filepath: Path to output file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            FileSummaryExporter._write_text_summary(f, stats)
        
        print(f"\n✅ Summary exported to: {filepath}")
    
    @staticmethod
    def export_to_json(stats: VideoStatistics, filepath: str) -> None:
        """
        Export summary to JSON file.
        
        Args:
            stats: Video statistics
            filepath: Path to output JSON file
        """
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "overview": {
                "total_frames": stats.total_frames,
                "persons_detected": stats.get_person_count(),
                "anomalies_detected": stats.get_anomaly_count()
            },
            "top_activities": [
                {"activity": activity, "count": count}
                for activity, count in stats.get_top_activities()
            ],
            "top_emotions": [
                {"emotion": emotion, "count": count}
                for emotion, count in stats.get_top_emotions()
            ],
            "persons": []
        }
        
        # Add per-person details
        for person in stats.get_sorted_persons():
            person_data = {
                "track_id": person.track_id,
                "frame_count": person.frame_count,
                "emotions": person.get_emotions_display(),
                "activities": person.get_activities_display(),
                "anomalies": [
                    {
                        "frame": anomaly.frame_number,
                        "explanation": anomaly.explanation
                    }
                    for anomaly in person.anomalies
                ],
                "movement_segments": [
                    {
                        "start_frame": seg.start_frame,
                        "end_frame": seg.end_frame,
                        "activity": seg.activity,
                        "emotion": seg.dominant_emotion,
                        "anomalies": seg.anomalies
                    }
                    for seg in person.movement_segments
                ]
            }
            summary_data["persons"].append(person_data)
        
        # Add all anomalies
        summary_data["all_anomalies"] = [
            {
                "frame": anomaly.frame_number,
                "person_id": anomaly.track_id,
                "explanation": anomaly.explanation
            }
            for anomaly in stats.all_anomalies
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ JSON summary exported to: {filepath}")
    
    @staticmethod
    def _write_text_summary(f: TextIO, stats: VideoStatistics) -> None:
        """
        Write formatted text summary to file handle.
        
        Args:
            f: File handle to write to
            stats: Video statistics
        """
        # Header
        f.write("=" * 80 + "\n")
        f.write(" " * 25 + "VIDEO ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overview Statistics
        f.write("📊 OVERVIEW STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Frames:       {stats.total_frames}\n")
        f.write(f"Persons Detected:   {stats.get_person_count()}\n")
        f.write(f"Anomalies Detected: {stats.get_anomaly_count()}\n\n")
        
        # Top Activities
        f.write("🏃 TOP ACTIVITIES\n")
        f.write("-" * 80 + "\n")
        top_activities = stats.get_top_activities()
        if top_activities:
            max_count = max(count for _, count in top_activities)
            for activity, count in top_activities:
                bar_length = int((count / max_count) * 40)
                bar = "█" * bar_length
                f.write(f"  {activity:<35} {count:>5}  {bar}\n")
        else:
            f.write("  No activity data available\n")
        f.write("\n")
        
        # Top Emotions
        f.write("😊 TOP EMOTIONS\n")
        f.write("-" * 80 + "\n")
        top_emotions = stats.get_top_emotions()
        if top_emotions:
            max_count = max(count for _, count in top_emotions)
            for emotion, count in top_emotions:
                bar_length = int((count / max_count) * 40)
                bar = "█" * bar_length
                f.write(f"  {emotion:<35} {count:>5}  {bar}\n")
        else:
            f.write("  No emotion data available\n")
        f.write("\n")
        
        # Per-Person Analysis
        if stats.get_person_count() > 0:
            f.write("👤 PER-PERSON ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            for person in stats.get_sorted_persons()[:20]:  # Limit to 20 persons
                f.write(f"Person ID #{person.track_id} ({person.frame_count} frames)\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Emotions:    {person.get_emotions_display()}\n")
                f.write(f"  Activities:  {person.get_activities_display()}\n")
                f.write(f"  Anomalies:   {len(person.anomalies)}\n")
                
                if person.anomalies:
                    f.write("\n  Anomaly Details:\n")
                    for anomaly in person.anomalies[:5]:  # Limit to 5 per person
                        f.write(f"    • Frame {anomaly.frame_number}: {anomaly.explanation}\n")
                
                if person.movement_segments:
                    f.write("\n  Movement Timeline:\n")
                    for seg in person.movement_segments[:10]:  # Limit to 10 segments
                        time_str = seg.get_duration_display()
                        anomaly_marker = " ⚠️" if seg.anomalies else ""
                        f.write(f"    • {time_str}: {seg.activity} ({seg.dominant_emotion}){anomaly_marker}\n")
                        if seg.anomalies:
                            for anom in seg.anomalies:
                                f.write(f"        ⚠️ {anom}\n")
                
                f.write("\n")
        
        # Anomalies Summary
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
            
            # Behavioral Anomalies
            if behavioral_anomalies:
                f.write("⚠️  BEHAVIORAL ANOMALIES\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total: {len(behavioral_anomalies)}\n\n")
                
                for anomaly in behavioral_anomalies[:30]:  # Limit to 30
                    f.write(f"  Frame {anomaly.frame_number:>5}, Person #{anomaly.track_id}: {anomaly.explanation}\n")
                
                if len(behavioral_anomalies) > 30:
                    f.write(f"\n  ... and {len(behavioral_anomalies) - 30} more behavioral anomalies\n")
                f.write("\n")
            
            # System Diagnostics
            if system_anomalies:
                f.write("🔧 SYSTEM DIAGNOSTICS (Tracking Errors)\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total: {len(system_anomalies)}\n\n")
                
                for anomaly in system_anomalies[:10]:  # Limit to 10
                    f.write(f"  Frame {anomaly.frame_number:>5}, Person #{anomaly.track_id}: {anomaly.explanation}\n")
                
                if len(system_anomalies) > 10:
                    f.write(f"\n  ... and {len(system_anomalies) - 10} more tracking errors\n")
                f.write("\n")
        
        # Footer
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "Summary generated successfully\n")
        f.write("=" * 80 + "\n")
