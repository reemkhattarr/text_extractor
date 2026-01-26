
import numpy as np
import cv2
import os
import sys

# Mock templates
# Create a dummy "C" image and a "1" image
def create_dummy_images():
    # White background, black text
    img_c = np.full((30, 20), 255, dtype=np.uint8)
    cv2.putText(img_c, "C", (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,), 2)
    
    img_1 = np.full((30, 10), 255, dtype=np.uint8)
    cv2.putText(img_1, "1", (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,), 2)
    
    return {"C": img_c, "1": img_1}

# We need to import match_character from matcher.py
# Make sure current dir is in path
sys.path.append(os.getcwd())
from matcher import match_character

def test_bias():
    templates = create_dummy_images()
    
    # CASE 1: Perfect C (0 deg)
    # create a candidate that is exactly C
    cand_c = templates["C"].copy()
    
    # Should match C at 0 deg
    char, score, angle = match_character(cand_c, templates)
    print(f"Test 1 (Perfect C): Got {char} at {angle} deg (Score: {score:.3f})")
    assert char == "C" and angle == 0, "Test 1 Failed"

    # CASE 2: Ambiguous Case
    # Create a candidate that matches "1" at 90 deg slightly better than "1" at 0 deg?
    # Hard to simulate synthetic better match at 90 deg without a specific shape.
    # Instead, we can verify that if we have a rotated candidate that matches 0 deg "well enough", it picks 0.
    
    # Let's take '1' and rotate it 90 degrees.
    # It looks like '-'.
    # If we only have '1' template (vertical), it won't match well.
    
    # CASE 3: Noisy Match
    # Let's just create a candidate that looks like 'C' but rotated 90 degrees (which looks like U).
    # And we ADD a 'U' template that is exactly that shape.
    
    # C template
    c_tmpl = templates["C"]
    
    # Create U template (C rotated 90 CCW)
    u_tmpl = cv2.rotate(c_tmpl, cv2.ROTATE_90_COUNTERCLOCKWISE)
    templates["U"] = u_tmpl
    
    # Input is U-shape (vertical C).
    cand_u = u_tmpl.copy()
    
    # Matcher:
    # 0 deg: Matches U match 1.0.
    # 90 deg (CW) -> Becomes C. Matches C match 1.0.
    
    # We have TIE: (U, 1.0, 0) and (C, 1.0, 90).
    # Bias should pick 0 deg -> U.
    char, score, angle = match_character(cand_u, templates)
    print(f"Test 2 (Tie U at 0 vs C at 90): Got {char} at {angle} deg (Score: {score:.3f})")
    assert angle == 0, "Test 2 Failed: Should prefer 0 deg on tie"
    assert char == "U", "Test 2 Failed: Should match U at 0 deg"
    
    print("\nAll Tests Passed!")

if __name__ == "__main__":
    try:
        test_bias()
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
