Python implementation code for the paper titled,

Title: 3D reconstruction of digital cores based on multi-discriminator GAN and improved residual blocks

Authors: Ting Zhang1, Mengling Ni1, Yi Du2, *, Anqin Zhang1, **

1.College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

2.School of Computer and Information Engineering, Institute for Artificial Intelligence, Shanghai Polytechnic University, Shanghai 201209, China

*Corresponding author: Yi Du (duyi0701@126.com) 

**Corresponding author: Anqin Zhang (bee921@yeah.net)

Ting Zhang Email: tingzh@shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Mengling Ni: y21108009@mail.shiep.edu.cn, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Anqin Zhang Email: bee921@yeah.net, Affiliation: College of Computer Science and Technology, Shanghai University of Electric Power, Shanghai 200090, China

Yi Du E-mail: duyi0701@126.com, Affiliation: School of Computer and Information Engineering, Institute for Artificial Intelligence, Shanghai Polytechnic University, Shanghai 201209, China

# MDGAN

1.requirements

Pytorch == 1.7.0

To run the code, an NVIDIA GeForce RTX3080 GPU video card with 10GB video memory is required.

Software development environment should be any Python integrated development environment used on an NVIDIA video card.

Programming language: Python 3.8.3.

2.How to use？

First, preprocess the image: Cut 80 digital core slices into 80 * 80 size images and stack sequentially into a tif of 80 * 80 * 80 size as a training image.

Secondly, set the network parameters such as maxsize, batchsize, learning rate and storage location. After configuring the parameters and environment, you can run main.py directly.
