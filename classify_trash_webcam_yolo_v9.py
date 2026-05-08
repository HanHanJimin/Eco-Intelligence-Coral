"""
Smart Trash Classification - Coral Edge TPU webcam demo.
For YOLOv8 classification model trained by team on
mostafaabla/garbage-classification (12 classes).

VERSION FOR FLOAT32 MODEL (best_final3_v9_edgetpu.tflite)
- Float32 input/output (simpler than int8 version)
- Standard 0-1 normalization
- Output is direct softmax probabilities

Usage:
    py -3.9 classify_trash_webcam_yolo_v9.py --model best_final3_v9_edgetpu.tflite

Controls:
    q          Quit
    SPACE      Save current frame
    f          Toggle center-crop / full-frame mode
"""

import argparse
import time
import os
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from pycoral.utils.edgetpu import make_interpreter


# 12 classes from mostafaabla/garbage-classification (alphabetical)
CLASS_NAMES = [
    'batteries',     # 0
    'biological',    # 1
    'brown-glass',   # 2
    'cardboard',     # 3
    'clothes',       # 4
    'green-glass',   # 5
    'metal',         # 6
    'paper',         # 7
    'plastic',       # 8
    'shoes',         # 9
    'trash',         # 10
    'white-glass',   # 11
]


# Team's 4-bin disposal scheme
BIN_PLASTIC  = {'name': 'PLASTIC BIN',     'color': (40, 40, 220)}    # Red (BGR)
BIN_METAL    = {'name': 'METAL BIN',       'color': (40, 220, 220)}   # Yellow
BIN_RECYCLE  = {'name': 'RECYCLE TRASH',   'color': (60, 200, 80)}    # Green
BIN_ORGANIC  = {'name': 'ORGANIC TRASH',   'color': (220, 140, 50)}   # Blue

CLASS_TO_BIN = {
    'plastic':     BIN_PLASTIC,
    'metal':       BIN_METAL,
    'paper':       BIN_RECYCLE,
    'cardboard':   BIN_RECYCLE,
    'green-glass': BIN_RECYCLE,
    'brown-glass': BIN_RECYCLE,
    'white-glass': BIN_RECYCLE,
    'clothes':     BIN_RECYCLE,
    'shoes':       BIN_RECYCLE,
    'batteries':   BIN_RECYCLE,
    'biological':  BIN_ORGANIC,
    'trash':       BIN_ORGANIC,
}


ACCENT_GREEN = (140, 220, 100)
ACCENT_AMBER = (60, 165, 245)
TEXT_PRIMARY = (245, 245, 245)
TEXT_MUTED   = (180, 180, 180)
TEXT_DIM     = (130, 130, 130)
BG_OVERLAY   = (20, 20, 20)


def get_bin(class_name):
    return CLASS_TO_BIN.get(class_name, BIN_ORGANIC)


def preprocess_float32(pil_img, input_size):
    """
    Preprocess for float32 model.
    Pixel values [0, 255] -> [0.0, 1.0] (standard normalization).
    """
    img = pil_img.resize(input_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def draw_overlay_panel(img, x, y, w, h, alpha=0.78, accent_color=None):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), BG_OVERLAY, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if accent_color is not None:
        cv2.rectangle(img, (x, y), (x + 3, y + h), accent_color, -1)


def draw_corner_brackets(img, x1, y1, x2, y2, color, thickness=3, length=24):
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2 - length, y1), (x2, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2 - length), (x1, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x2 - length, y2), (x2, y2), color, thickness)
    cv2.line(img, (x2, y2 - length), (x2, y2), color, thickness)


def draw_progress_bar(img, x, y, w, h, fraction, fill_color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill_w = int(w * max(0.0, min(1.0, fraction)))
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), fill_color, -1)


def draw_bin_swatch(img, x, y, size, color):
    cv2.rectangle(img, (x, y), (x + size, y + size), color, -1)
    cv2.rectangle(img, (x, y), (x + size, y + size), TEXT_DIM, 1)
    inner_pad = 6
    cv2.rectangle(img,
                  (x + inner_pad, y + inner_pad),
                  (x + size - inner_pad, y + size - inner_pad),
                  (max(0, color[0] - 40), max(0, color[1] - 40), max(0, color[2] - 40)),
                  1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--save-dir', default='saved_frames_yolo_v9')
    parser.add_argument('--crop-size', type=int, default=480)
    args = parser.parse_args()

    print('Loading YOLOv8 classification model (float32 version)...')
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details['shape']
    input_size = (input_shape[2], input_shape[1])
    print(f'Input shape: {input_shape}, dtype: {input_details["dtype"].__name__}')
    print(f'Output shape: {output_details["shape"]}, dtype: {output_details["dtype"].__name__}')
    print(f'Number of classes: {output_details["shape"][1]}')
    print(f'Classes: {CLASS_NAMES}')

    print()
    print('Bin assignments:')
    for cls in CLASS_NAMES:
        b = get_bin(cls)
        print(f'  {cls:14s} -> {b["name"]}')

    os.makedirs(args.save_dir, exist_ok=True)

    print(f'\nOpening camera {args.camera}...')
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f'ERROR: Cannot open camera {args.camera}')
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print('\nCamera ready.')
    print('  q     = quit')
    print('  SPACE = save frame')
    print('  f     = toggle center-crop / full-frame\n')
    print('TIP: Hold item to FILL most of the green box for best accuracy.')
    print('     Plain background works much better than cluttered scenes.\n')

    prediction_buffer = deque(maxlen=10)
    fps_buffer = deque(maxlen=30)
    last_time = time.time()
    saved_count = 0
    use_center_crop = True
    pulse_t = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        crop_size = min(args.crop_size, h, w)
        cx, cy = w // 2, h // 2
        x1 = cx - crop_size // 2
        y1 = cy - crop_size // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        if use_center_crop:
            classify_region = frame[y1:y2, x1:x2]
        else:
            classify_region = frame

        rgb = cv2.cvtColor(classify_region, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        # Float32 preprocessing - just normalize to 0-1
        input_data = preprocess_float32(pil_img, input_size)
        input_data = np.expand_dims(input_data, axis=0)

        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()

        # Output is float32 probabilities directly - no dequantization needed
        probs = interpreter.get_tensor(output_details['index'])[0]
        probs = np.clip(probs, 0.0, 1.0)

        top_idx = int(np.argmax(probs))
        top_score = float(probs[top_idx])
        top_label = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f'class_{top_idx}'

        prediction_buffer.append((top_label, top_score))
        recent_labels = [p[0] for p in prediction_buffer]
        smoothed_label = Counter(recent_labels).most_common(1)[0][0]
        matching = [s for l, s in prediction_buffer if l == smoothed_label]
        smoothed_score = sum(matching) / len(matching)

        bin_info = get_bin(smoothed_label)

        now = time.time()
        fps_buffer.append(now - last_time)
        last_time = now
        fps = len(fps_buffer) / sum(fps_buffer) if sum(fps_buffer) > 0 else 0
        pulse_t += 0.1

        confident = smoothed_score >= args.threshold
        accent = ACCENT_GREEN if confident else ACCENT_AMBER

        # Top bar
        top_bar = frame[0:60, :].copy()
        cv2.rectangle(top_bar, (0, 0), (w, 60), BG_OVERLAY, -1)
        cv2.addWeighted(top_bar, 0.6, frame[0:60, :], 0.4, 0, frame[0:60, :])

        pulse_alpha = 0.6 + 0.4 * abs(np.sin(pulse_t))
        dot_color = (int(140 * pulse_alpha + 60), int(220 * pulse_alpha + 30), int(100 * pulse_alpha + 30))
        cv2.circle(frame, (28, 30), 5, dot_color, -1)
        cv2.putText(frame, 'ECO-INTELLIGENCE  YOLO v9', (45, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (159, 225, 203), 1, cv2.LINE_AA)

        mode_text = 'CENTER' if use_center_crop else 'FULL'
        status_items = [('FPS', f'{fps:.1f}'),
                        ('SAVED', str(saved_count)),
                        ('MODE', mode_text)]
        right_x = w - 20
        for label_text, value_text in reversed(status_items):
            value_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            block_w = value_size[0] + label_size[0] + 10
            cv2.putText(frame, value_text,
                        (right_x - value_size[0], 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(frame, label_text,
                        (right_x - value_size[0] - label_size[0] - 8, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
            right_x -= block_w + 18

        if use_center_crop:
            draw_corner_brackets(frame, x1, y1, x2, y2, accent, thickness=3, length=28)

        # Detected card
        card_x = 20
        card_y = h - 200
        card_w = 320
        card_h = 100
        draw_overlay_panel(frame, card_x, card_y, card_w, card_h,
                           alpha=0.82, accent_color=accent)

        cv2.putText(frame, 'DETECTED', (card_x + 16, card_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
        display_label = smoothed_label.upper() if confident else 'UNCERTAIN'
        cv2.putText(frame, display_label, (card_x + 16, card_y + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, accent, 2, cv2.LINE_AA)

        bar_x = card_x + 16
        bar_y = card_y + 75
        bar_w = 180
        bar_h = 4
        draw_progress_bar(frame, bar_x, bar_y, bar_w, bar_h, smoothed_score, accent)
        cv2.putText(frame, f'{smoothed_score:.2f}',
                    (bar_x + bar_w + 12, bar_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUTED, 1, cv2.LINE_AA)

        if confident:
            bin_card_w = 320
            bin_card_h = 100
            bin_card_x = w - bin_card_w - 20
            bin_card_y = h - 200
            draw_overlay_panel(frame, bin_card_x, bin_card_y, bin_card_w, bin_card_h, alpha=0.82)

            cv2.putText(frame, 'DISPOSE IN', (bin_card_x + 16, bin_card_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)

            swatch_size = 42
            swatch_x = bin_card_x + 16
            swatch_y = bin_card_y + 36
            draw_bin_swatch(frame, swatch_x, swatch_y, swatch_size, bin_info['color'])

            text_x = swatch_x + swatch_size + 14
            cv2.putText(frame, bin_info['name'], (text_x, bin_card_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bin_info['color'], 2, cv2.LINE_AA)

        hint_y = h - 24
        hints = [('Q', 'quit'), ('SPACE', 'save'), ('F', 'mode')]
        sizes = []
        total_width = 0
        for key, action in hints:
            key_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            action_size = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            block = key_size[0] + 16 + action_size[0]
            sizes.append((key_size, action_size, block))
            total_width += block + 30
        total_width -= 30

        hx = (w - total_width) // 2
        for (key, action), (key_size, action_size, block) in zip(hints, sizes):
            pill_x = hx
            pill_y = hint_y - 14
            pill_w = key_size[0] + 12
            pill_h = 18
            cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h),
                          (50, 50, 50), -1)
            cv2.putText(frame, key, (pill_x + 6, pill_y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(frame, action, (pill_x + pill_w + 8, hint_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_MUTED, 1, cv2.LINE_AA)
            hx += block + 30

        cv2.imshow('Eco-Intelligence YOLO v9 - Coral Edge TPU', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('Quitting...')
            break
        elif key == ord(' '):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            safe_label = smoothed_label.replace(' ', '_').replace('/', '_')
            filename = f'{timestamp}_{safe_label}_{smoothed_score:.2f}.jpg'
            filepath = os.path.join(args.save_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            print(f'[{saved_count}] Saved: {filename} ({bin_info["name"]})')
        elif key == ord('f'):
            use_center_crop = not use_center_crop
            print(f'Mode: {"CENTER-CROP" if use_center_crop else "FULL-FRAME"}')

    cap.release()
    cv2.destroyAllWindows()
    print(f'\nSession ended. Saved {saved_count} frames to {args.save_dir}/')


if __name__ == '__main__':
    main()
