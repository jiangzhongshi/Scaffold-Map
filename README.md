# Simplicial Complex Augmentation Framework for Bijective Maps

[Zhongshi Jiang](http://cs.nyu.edu/~zhongshi/), [Scott Schaefer](http://faculty.cs.tamu.edu/schaefer/), [Daniele Panozzo](http://cs.nyu.edu/~panozzo/)<br/>
*ACM Transaction on Graphics (Proceedings of SIGGRAPH Asia 2017)*<br/>
DOI: 10.1145/3130800.3130895
## Abstract
Bijective maps are commonly used in many computer graphics and scientific computing applications, including texture, displacement, and bump mapping. However, their computation is numerically challenging due to the global nature of the problem, which makes standard smooth optimization techniques prohibitively expensive.
We propose to use a scaffold structure to reduce this challenging and global problem to a local injectivity condition. This construction allows us to benefit from the recent advancements in locally injective maps optimization to efficiently compute large scale bijective maps (both in 2D and 3D), sidestepping the need to explicitly detect and avoid collisions.
Our algorithm is guaranteed to robustly compute a globally bijective map, both in 2D and 3D. To demonstrate the practical applicability, we use it to compute globally bijective single patch parametrizations, to pack multiple charts into a single UV domain, to remove self-intersections from existing models, and to deform 3D objects while preventing self-intersections.
Our approach is simple to implement, efficient (two orders of magnitude faster than competing methods), and robust, as we demonstrate in a stress test on a parametrization dataset with over a hundred meshes.

## Source Code
Source code is hosted on this GitHub repository. The program is built and tested on MacOS system with AppleClang.

### Downloading
```bash
git clone --recursive https://github.com/jiangzhongshi/scaffold-map.git
```
### Building [![Build Status](https://travis-ci.org/jiangzhongshi/scaffold-map.svg)](https://travis-ci.org/jiangzhongshi/scaffold-map)

To build the program, you can use the CMakeLists.txt in
the root folder:

```bash
mkdir build
cd build
cmake ../
make
```
### Running
```bash
./build/scaffold-test_bin models/camel_b.obj
```

## License
[MPL2](http://www.mozilla.org/MPL/2.0/) licensed
([FAQ](http://www.mozilla.org/MPL/2.0/FAQ.html))
