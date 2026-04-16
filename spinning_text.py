#!/usr/bin/env python3
"""
Spinning 3D text shape — asks for a password, then renders a rotating
torus (donut) built entirely out of the password's letter characters.
"""

import sys
import os
import math
import time
import getpass

# Torus geometry parameters
R1 = 1.0   # radius of the tube cross-section
R2 = 2.0   # distance from the centre of the tube to the centre of the torus
K2 = 5.0   # distance from the viewer to the torus
SCREEN_W = 80
SCREEN_H = 40
K1 = SCREEN_W * K2 * 3 / (8 * (R1 + R2))  # projection constant

LUMINANCE_CHARS = ".,-~:;=!*#$@"  # brightness ramp (dark → bright)


def render_frame(A: float, B: float, letters: str) -> str:
    """Return one frame of the spinning torus as a string."""
    output = [[" "] * SCREEN_W for _ in range(SCREEN_H)]
    zbuffer = [[0.0] * SCREEN_W for _ in range(SCREEN_H)]

    cos_A, sin_A = math.cos(A), math.sin(A)
    cos_B, sin_B = math.cos(B), math.sin(B)

    letter_idx = 0  # walks through the password characters

    theta = 0.0
    while theta < 2 * math.pi:
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        phi = 0.0
        while phi < 2 * math.pi:
            cos_p, sin_p = math.cos(phi), math.sin(phi)

            # 3-D coordinates of the point on the torus surface
            cx = R2 + R1 * cos_t
            x = cx * (cos_B * cos_p + sin_A * sin_B * sin_p) - R1 * cos_A * sin_B * sin_t
            y = cx * (sin_B * cos_p - sin_A * cos_B * sin_p) + R1 * cos_A * cos_B * sin_t
            z = K2 + cos_A * cx * sin_p + R1 * sin_A * sin_t
            ooz = 1.0 / z  # one-over-z (for depth)

            # project to 2-D screen
            xp = int(SCREEN_W / 2 + K1 * ooz * x)
            yp = int(SCREEN_H / 2 - K1 * ooz * y * 0.5)

            # simple directional lighting
            L = (cos_t * cos_p * sin_B
                 - cos_A * sin_t * cos_p
                 - sin_A * sin_p
                 + cos_B * (cos_A * sin_t - cos_t * sin_A * sin_p))

            if 0 <= xp < SCREEN_W and 0 <= yp < SCREEN_H and ooz > zbuffer[yp][xp]:
                zbuffer[yp][xp] = ooz
                # Pick a character from the password based on surface position,
                # but modulate its brightness via luminance mapping
                lum_index = max(0, int(L * 8))
                if lum_index > 0:
                    # Use password letters for bright regions
                    ch = letters[letter_idx % len(letters)]
                    letter_idx += 1
                else:
                    # Dim regions get a subtle dot
                    ch = "."
                output[yp][xp] = ch

            phi += 0.07
        theta += 0.07

    return "\n".join("".join(row) for row in output)


def main() -> None:
    password = getpass.getpass("Enter password to unlock the animation: ")

    if not password:
        print("No password entered. Exiting.")
        sys.exit(1)

    # Use only the letter characters from the password for the shape
    letters = [ch for ch in password if ch.isalpha()]
    if not letters:
        # Fall back to the full password if it has no letters
        letters = list(password)

    print(f"\nSpinning a donut made of your password's letters!  (Ctrl+C to quit)\n")
    time.sleep(1)

    A = 0.0
    B = 0.0

    try:
        while True:
            frame = render_frame(A, B, letters)
            # Move cursor to top-left and draw
            sys.stdout.write("\x1b[H\x1b[2J")  # clear screen
            sys.stdout.write(frame)
            sys.stdout.flush()

            A += 0.04
            B += 0.02
            time.sleep(0.03)
    except KeyboardInterrupt:
        # Restore terminal on exit
        sys.stdout.write("\x1b[2J\x1b[H")
        print("Goodbye!")


if __name__ == "__main__":
    main()
