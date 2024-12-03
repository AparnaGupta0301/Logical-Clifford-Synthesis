# Logical Clifford Synthesis for Stabilizer Codes

This repository contains implementations of algorithms for synthesizing logical Clifford operators for stabilizer codes in quantum computation. These algorithms provide a systematic framework for translating logical operations into physical quantum circuits.

## Overview of Algorithms

### Algorithm 1: Finding a Symplectic Matrix for Linear Systems
- **Purpose**: Solve a system of linear equations in the symplectic group \( Sp(2m, F_2) \) using symplectic transvections.
- **Steps**:
  1. Evaluate the symplectic inner product of row vectors to determine transformations.
  2. Apply necessary symplectic transvections for corrections.
  3. Iterate over vectors to ensure they map correctly, preserving the symplectic structure.
- **Implementation**: `algorithm_1.py`

---

### Algorithm 2: Enumerating All Symplectic Solutions
- **Purpose**: Enumerates all symplectic solutions for a system of linear equations by leveraging degrees of freedom in partially constrained matrices.
- **Steps**:
  1. Use **Algorithm 1** to find a particular solution to the system.
  2. Identify free vectors in the symplectic basis that can vary while maintaining valid symplectic constraints.
  3. Enumerate all combinations of free vectors to generate all solutions.
- **Implementation**: `algorithm_2.py`

---

### Algorithm 3: Logical Clifford Synthesis (LCS) [Under Development]
- **Purpose**: Synthesizes all physical realizations of logical Clifford operators for stabilizer codes.
- **Steps**:
  1. Translate logical Clifford operations into linear constraints on symplectic matrices.
  2. Use **Algorithm 2** to find all symplectic solutions.
  3. Translate symplectic matrices into physical quantum circuits.
- **Implementation**: Currently under development. The file will is named `algorithm_3.py`.Here's the revised version with improved grammar and clarity:


Refer to the example notebooks in the `Examples` folder for implementation examples. Please note that the example notebook for **Algorithm 3** is incomplete and still under development. The author is also actively working on providing comprehensive documentation for all usage scenarios. 

---
## **Citation**
If you use these algorithms in your research, please cite the related publication:

Rengaswamy, Narayanan; Calderbank, Robert; Kadhe, Swanand; and Pfister, Henry D.  
"Logical Clifford Synthesis for Stabilizer Codes."  
*IEEE Transactions on Quantum Engineering*, vol. 1, 2020, pp. 1–23.  
DOI: [10.1109/TQE.2020.3023419](https://doi.org/10.1109/TQE.2020.3023419)

---
## **MIT License**

Copyright (c) 2024 Aparna Gupta, University of Arizona

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
