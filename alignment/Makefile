# The Manim executable and script
MANIM = manim
SCRIPT = scene.py

# Manim quality options (you can change these as needed)
QUALITY = -pql  # Low quality for fast rendering, change to -pqh for high quality

# Scene targets
SCENES = Alignment AlignmentDp Msa

# Default target to build all scenes
all: $(SCENES)

# Individual targets for each scene
Alignment: $(SCRIPT)
	$(MANIM) $(QUALITY) $< $@

AlignmentDp: $(SCRIPT)
	$(MANIM) $(QUALITY) $< $@

Msa: $(SCRIPT)
	$(MANIM) $(QUALITY) $< $@

# Clean up generated files
clean:
	rm -f media/videos/*.mp4
	rm -f media/images/*.png

.PHONY: all clean $(SCENES)
