"""Decompose a Dyn-Bench question into sub-problems, solve them, assemble the answer.

The decision graph (see ``testing_decomposition.ipynb``, first cell):

S0  clip context -- the generator's frame list, its time axis, and (after the GPU prep)
    the Point4D scene pass: camera pose and intrinsics per frame, depth, gravity, scale.
S1  decompose -- question + options -> a typed ``QueryProgram`` (kind first, then slots).
S2  ground -- every entity role -> SAM3 instances -> the role's instance(s).
S3  measure -- lift masks through Point4D depth -> signal in a frame and window -> reduce.
S4  interpret -- score every option against the reduced measurements -> leaf A / B / C.

Modules:

``scene4d``   the per-clip state (M1: generator frames and time axis)
``program``   the Query Program, the kinds registry, option vocabularies, the rule parser
``build_catalog``  CLI that resolves the catalog questions and their clips on disk
"""
