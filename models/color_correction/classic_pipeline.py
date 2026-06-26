import torch


class ClassicCorrectionPipeline():

    def __call__(self, rgb, ill, ill_cct, cct_2500, cct_6500):
        return self.correct(rgb, ill, ill_cct, cct_2500, cct_6500)

    def correct(self, rgb, ill, ill_cct, cct_2500, cct_6500):
        """
        Apply color correction to RGB image.
        
        Args:
            rgb: Input RGB image (B, 3, H, W)
            ill: Illuminant - either (B, 3) triplet or (B, 3, H, W) 2D map
            ill_cct: CCT conversion matrix (B, 3, 3)
            cct_2500: CCM at 2500K (B, 3, 3)
            cct_6500: CCM at 6500K (B, 3, 3)
        
        Returns:
            Corrected RGB image (B, 3, H, W)
        """
        b, c, h, w = rgb.shape
        
        return self.correct_global(rgb, ill, ill_cct, cct_2500, cct_6500)
    
    def correct_global(self, rgb, ill, ill_cct, cct_2500, cct_6500):
        """Standard correction with a single illuminant per image."""
        b, c, h, w = rgb.shape
        
        ill = ill / torch.norm(ill + 1e-9, dim=1, keepdim=True)

        ccm = self.interpolate_ccms(ill, cct_2500, cct_6500, ill_cct)
        
        corrected = torch.einsum('bmn,bnl->bml', ccm.transpose(1, 2), (rgb / (ill[:, :, None, None] + 1e-8)).reshape(b, c, -1)).reshape(b, c, h, w)

        return corrected.clamp(0, 1)
       
    def CCT_McCamy(self, xyzs):
        """
        Correlated Color Temperature (CCT) calculator using McCamy's method.
        Based on xy to CCT conversion.

        Args:
            xyzs: Nx3 array of CIE XYZ values

        n = (x-0.3320)/(0.1858-y);
        CCT = 437*n^3 + 3601*n^2 + 6861*n + 5517
        """

        Xs, Ys, Zs = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

        xs = Xs / (Xs + Ys + Zs + 1e-9)
        ys = Ys / (Xs + Ys + Zs + 1e-9)

        ns = (xs - 0.3320) / (0.1858 - ys + 1e-9)

        ccts = 437 * ns**3 + 3601 * ns**2 + 6861 * ns + 5517

        status = torch.where((ccts < 1667) | (ccts > 25000), -1, 0)  # McCamy's formula is valid only in this range

        return status, ccts

    def interpolate_ccms(self, ill_rgbs, cct1, cct2, ill_cct, temp1=2500, temp2=6500):

        # Correlated color temperature of the illuminant
        xyzs = torch.einsum('bmn,bn->bm', ill_cct, ill_rgbs)

        status, ccts = self.CCT_McCamy(xyzs)
        
        # if torch.any(status != 0):
        #     print("Warning: CCT calculation failed for some samples!")
            
        gs = (ccts**(-1) - temp2**(-1)) / (temp1**(-1) - temp2**(-1))
        gs = gs[:, None, None]

        ccms = gs * cct1 + (1 - gs) * cct2

        return ccms
