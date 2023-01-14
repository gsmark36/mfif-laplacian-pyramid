# Multi-focus image fusion based on Laplacian pyramid
Packages used: **numpy**, **skimage**, **matplotlib**.  
The algorithm follows the steps:  
•	Decompose two input images into Laplacian pyramids.  
•	Generate the top level of fused pyramid by maximum region information rule.  
•	Generate other levels by maximum region energy rule.  
•	Reconstruct fused image by performing inverse transformation to fused Laplacian pyramid.  
## References
[1]	Wang, Wencheng, and Faliang Chang. "A Multi-focus Image Fusion Method Based on Laplacian Pyramid." J. Comput. 6.12 (2011): 2559-2566.  
[2]	P. Burt and E. Adelson, "The Laplacian Pyramid as a Compact Image Code," in IEEE Transactions on Communications, vol. 31, no. 4, pp. 532-540, April 1983, doi: 10.1109/TCOM.1983.1095851.  
