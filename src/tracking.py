import os
import argparse

# Configurar variáveis de ambiente
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

from ultralytics import YOLO

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='YOLO Face Mask Detection with Tracker Selection'
    )

    # Argumentos mutuamente exclusivos para escolher o tracker
    tracker_group = parser.add_mutually_exclusive_group()
    tracker_group.add_argument(
        '-bytetrack',
        '--bytetrack',
        action='store_true',
        help='Use ByteTrack tracker (default)'
    )
    tracker_group.add_argument(
        '-botsort',
        '--botsort',
        action='store_true',
        help='Use BotSORT tracker'
    )

    # Outros argumentos opcionais
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source (0 for webcam, path for video file, URL for stream)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.5,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (cuda, cpu, 0, 1, etc.)'
    )
    parser.add_argument(
        '--no-show',
        action='store_false',
        dest='show',
        help='Disable showing the video output'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save the output video'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='../model/face_mask_yolo/weights/best.pt',
        help='Path to model weights'
    )

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()

    # Determine tracker based on arguments
    if args.botsort:
        tracker_name = "botsort.yaml"
        tracker_display = "BotSORT"
    else:
        # Default to ByteTrack if no tracker specified
        tracker_name = "bytetrack.yaml"
        tracker_display = "ByteTrack"

    print(f"Using tracker: {tracker_display}")
    print(f"Source: {args.source}")
    print(f"Confidence threshold: {args.conf}")

    # Load model
    try:
        model = YOLO(args.model_path)
        print(f"Model loaded successfully from: {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Configure device if specified
    if args.device:
        model.to(args.device)
        print(f"Using device: {args.device}")

    # Convert source to int if it's a webcam number
    try:
        source = int(args.source) if args.source.isdigit() else args.source
    except ValueError:
        source = args.source

    # Run tracking
    try:
        model.track(
            source=source,
            show=args.show,
            tracker=tracker_name,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            verbose=False,
            persist=True,
            device=args.device,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"Error during tracking: {e}")

if __name__ == "__main__":
    main()