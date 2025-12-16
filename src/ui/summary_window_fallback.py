    def _print_basic_summary(self, stats: VideoStatistics) -> None:
        """Print basic text summary when Rich fails."""
        print("\n" + "="*80)
        print(" "*25 + "VIDEO ANALYSIS SUM MARY")
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
            print(f"\n--- Anomalies ({len(stats.all_anomalies)} total, showing first 10) ---")
            for anomaly in stats.all_anomalies[:10]:
                print(f"  Frame {anomaly.frame_number}, ID #{anomaly.track_id}: {anomaly.explanation}")
        
        print("\n" + "="*80)
        print(" "*20 + "Summary generated successfully")
        print("="*80 + "\n")
