# DistillGenBioPath

*Remaining files and weights will be available soon.*


The goal of this project is to distill knowledge from GenBio-PathFM (a large foundation model for histopathology) into a smaller and more efficient/specialized model KAiko ViT-S/16. Data is biopsy whole-slide images (WSIs) in DICOM format. The idea is to transfer the knowledge learned by the larger model to KAiko while keeping it lightweight and computationally efficient and moreover to have a smaller model runnable on any computers (considering pathologists are unequipped of GPUs or equivalent) and have specialized models for a specific disease/biopsy study. The resulting model can be used as a feature extractor inside an ML pipeline.
