# The Manim executable and script
MANIM = manim
SCRIPT = scene.py

# Manim quality options (you can change these as needed)
Q ?= l  # Low quality for fast rendering, change to -pqh for high quality

# Scene targets
SCENES = Alignment AlignmentDp Msa

# Default target to build all scenes
all: $(SCENES)

# Individual targets for each scene
Alignment: $(SCRIPT)
	$(MANIM) -pq$(Q) $< $@

AlignmentDp: $(SCRIPT)
	$(MANIM) -pq$(Q) $< $@

Msa: $(SCRIPT)
	$(MANIM) -pq$(Q) $< $@

Dist: $(SCRIPT)
	$(MANIM) -pq$(Q) $< $@


# Clean up generated files
clean:
	rm -f media/videos/*.mp4
	rm -f media/images/*.png

.PHONY: all clean $(SCENES)
