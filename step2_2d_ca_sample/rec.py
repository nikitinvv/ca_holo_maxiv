import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import cupyx.scipy.ndimage as ndimage
import os
from cuda_kernels import *
from utils import *
from chunking import gpu_batch
import time

cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)

class Rec:
    def __init__(self, args):
        for key, value in vars(args).items():
            setattr(self, key, value)       

        # allocate gpu and pinned memory linear_batch
        # calculate max buffer size
        nbytes = 32 * self.nchunk * self.npatch **2 * np.dtype("complex64").itemsize            
        
        # create CUDA streams and allocate pinned memory
        self.stream = [[[] for _ in range(3)] for _ in range(self.ngpus)]
        self.pinned_mem = [[] for _ in range(self.ngpus)]
        self.gpu_mem = [[] for _ in range(self.ngpus)]
        self.fker = [[] for _ in range(self.ngpus)]
        self.fkerc = [[] for _ in range(self.ngpus)]
        self.pool_inp = [[] for _ in range(self.ngpus)]
        self.pool_out = [[] for _ in range(self.ngpus)]
        self.pool = ThreadPoolExecutor(self.ngpus)    
        
        for igpu in range(self.ngpus):
            with cp.cuda.Device(igpu):
                self.pinned_mem[igpu] = cp.cuda.alloc_pinned_memory(nbytes)
                self.gpu_mem[igpu] = cp.cuda.alloc(nbytes)
                for k in range(3):
                    self.stream[igpu][k] = cp.cuda.Stream(non_blocking=False)
                fx = cp.fft.fftfreq(self.npsi * 2, d=self.voxelsize).astype("float32")
                [fx, fy] = cp.meshgrid(fx, fx)
                self.fker[igpu] = cp.exp(-1j * cp.pi * self.wavelength * self.distance * (fx**2 + fy**2)).astype('complex64')
                self.fkerc[igpu] = cp.exp(-1j * cp.pi * self.wavelength * self.distancec * (fx**2 + fy**2)).astype('complex64')
                self.pool_inp[igpu] = ThreadPoolExecutor(32 // self.ngpus)    
                self.pool_out[igpu] = ThreadPoolExecutor(32 // self.ngpus)
        
        self.pool_cpu = ThreadPoolExecutor(32) 


    def redot_batch(self,x,y):
        flg = chunking_flg(locals().values())        
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _redot(self, res, x,y):
            res[:] += redot(x, y)
            return res
        _redot(self, res, x,y)
        if flg:
            for k in range(1,len(res)):
                res[0] += res[k]
            res = res[0]    
        res = res[0]    
        return res
    

    def linear_batch(self,x,y,a,b):
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty_like(x)
        else:
            res = cp.empty_like(x)
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _linear(self, res, x,y,a,b):
            res[:] = a*x[:]+b*y[:]
        _linear(self, res, x,y,a,b)
        return res
    
    def mulc_batch(self,x,a):
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty_like(x)
        else:
            res = cp.empty_like(x)
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _mulc(self, res, x,a):
            res[:] = a*x[:]
        _mulc(self, res, x,a)
        return res
    
    def BH(self, d, vars):
        d = np.sqrt(d)
        
        alpha = 1
        rho = self.rho
        reused = {}

        for i in range(self.niter):
            # Calc reused variables and gradF
            self.calc_reused(reused, vars)
            
            self.gradientF(reused, d)
            
            # debug and visualization
            self.error_debug(vars, reused, d, i)
            
            self.vis_debug(vars, i)

            # gradients for each variable
            grads = self.gradients(vars, reused)
            
            if i == 0:
                etas = {}
                etas["psi"] = -grads["psi"] * rho[0] ** 2
                etas["q"] = -grads["q"] * rho[1] ** 2
                etas["r"] = -grads["r"] * rho[2] ** 2
            else:
                # conjugate direction
                beta = self.calc_beta(vars, grads, etas, reused, d)
                etas["psi"] = -grads["psi"] * rho[0] ** 2 + beta * etas["psi"]
                etas["q"] = -grads["q"] * rho[1] ** 2 + beta * etas["q"]
                etas["r"] = -grads["r"] * rho[2] ** 2 + beta * etas["r"]

            # step length
            alpha, top, bottom = self.calc_alpha(vars, grads, etas, reused, d)
            # print(alpha,top,bottom)
            # debug approxmation
            self.plot_debug(vars, etas, top, bottom, alpha, d, i)
            
            vars["psi"] += alpha * etas["psi"]
            vars["q"] += alpha * etas["q"]
            vars["r"] += alpha * etas["r"]

        return vars
    
    
    def E(self, ri, code):
        """Extract patches"""

        flg = chunking_flg(locals().values())
        # memory for result
        if flg:
            patches = np.empty([len(ri), self.npatch, self.npatch], dtype="complex64")
        else:
            patches = cp.empty([len(ri), self.npatch, self.npatch], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _E(self, patches, ri, code):
            stx = self.ncode // 2 - ri[:, 1] - self.npatch // 2
            sty = self.ncode // 2 - ri[:, 0] - self.npatch // 2
            Efast_kernel(
                (
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(len(ri) / 4)),
                ),
                (16, 16, 4),
                (patches, code, stx, sty, len(ri), self.npatch, self.ncode),
            )

        _E(self, patches, ri, code)

        return patches

    def ET(self, patches, ri):
        """Place patches, note only on 1 gpu for now"""
        
        flg = chunking_flg(locals().values())        
        if flg:
            code = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    code.append(cp.zeros([self.ncode, self.ncode], dtype="complex64"))
        else:
            code = cp.zeros([self.ncode, self.ncode], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _ET(self, code, patches, ri):
            """Adjoint extract patches"""
            stx = self.ncode // 2 - ri[:, 1] - self.npatch // 2
            sty = self.ncode // 2 - ri[:, 0] - self.npatch // 2
            ETfast_kernel(
                (
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(self.npatch / 16)),
                    int(cp.ceil(len(ri) / 4)),
                ),
                (16, 16, 4),
                (patches, code, stx, sty, len(ri), self.npatch, self.ncode),
            )

        _ET(self, code, patches, ri)
        if flg:
            for k in range(1,len(code)):
                code[0] += code[k]
            code = code[0]    
        return code

    def S(self, ri, r, code):
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            patches = np.empty([len(ri), self.npsi, self.npsi], dtype="complex64")
        else:
            patches = cp.empty([len(ri), self.npsi, self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _S(self, patches, ri, r, code):
            """Extract patches with subpixel shift"""
            coder = self.E(ri, code)

            x = cp.fft.fftfreq(self.npatch).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(
                -2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])
            ).astype("complex64")
            coder = cp.fft.ifft2(tmp * cp.fft.fft2(coder))
            patches[:] = coder[
                :, self.ex : self.npatch - self.ex, self.ex : self.npatch - self.ex
            ]

        _S(self, patches, ri, r, code)
        return patches

    def ST(self, patches, ri, r):
        """Place patches, note only on 1 gpu for now"""

        flg = chunking_flg(locals().values())        
        if flg:
            code = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    code.append(cp.zeros([self.ncode, self.ncode], dtype="complex64"))
        else:
            code = cp.zeros([self.ncode, self.ncode], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _ST(self, code, patches , ri, r):
            """Adjont extract patches with subpixel shift"""
            coder = cp.pad(patches, ((0, 0), (self.ex, self.ex), (self.ex, self.ex)))

            x = cp.fft.fftfreq(self.npatch).astype("float32")
            [y, x] = cp.meshgrid(x, x)
            tmp = cp.exp(
                2 * cp.pi * 1j * (y * r[:, 1, None, None] + x * r[:, 0, None, None])
            ).astype("complex64")
            coder = cp.fft.ifft2(tmp * cp.fft.fft2(coder))

            code[:] += self.ET(coder, ri)

        _ST(self, code, patches, ri, r)
        if flg:
            for k in range(1,len(code)):
                code[0] += code[k]
            code = code[0]    
        return code

    

    def _fwd_pad(self,f):
        """Fwd data padding"""
        [ntheta, n] = f.shape[:2]
        fpad = cp.zeros([ntheta, 2*n, 2*n], dtype='complex64')
        pad_kernel((int(cp.ceil(2*n+2/32)), int(cp.ceil(2*n/32)), ntheta),
                (32, 32, 1), (fpad, f, n, ntheta, 0))
        return fpad/2
    
    def _adj_pad(self, fpad):
        """Adj data padding"""
        [ntheta, n] = fpad.shape[:2]
        n //= 2
        f = cp.zeros([ntheta, n, n], dtype='complex64')
        pad_kernel((int(cp.ceil(2*n/32)), int(cp.ceil(2*n/32)), ntheta),
                (32, 32, 1), (fpad, f, n, ntheta, 1))
        return f/2
    
    def D(self, psi):
        """Forward propagator"""
 
        flg = chunking_flg(locals().values())        
        if flg:
            big_psi = np.empty([len(psi), self.n, self.n], dtype="complex64")
        else:
            big_psi = cp.empty([len(psi), self.n, self.n], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _D(self, big_psi, psi):
            ff = psi.copy()
            ff = self._fwd_pad(ff)
            v = cp.ones(2*self.npsi,dtype='float32')
            v[:self.npsi//2] = cp.sin(cp.linspace(0,1,self.npsi//2)*cp.pi/2)
            v[-self.npsi//2:] = cp.cos(cp.linspace(0,1,self.npsi//2)*cp.pi/2)
            v = cp.outer(v,v)
            ff *= v*2        
            ff = cp.fft.ifft2(cp.fft.fft2(ff)*self.fker[cp.cuda.Device().id])
            ff = ff[:,self.npsi//2:-self.npsi//2,self.npsi//2:-self.npsi//2]
            
            big_psi[:] = ff[:, self.pad : self.npsi - self.pad, self.pad : self.npsi - self.pad]
            
        
        _D(self, big_psi, psi)
        return big_psi


    def DT(self, big_psi):
        """Adjoint propagator"""

        flg = chunking_flg(locals().values())        
        if flg:
            psi = np.empty([len(big_psi), self.npsi, self.npsi], dtype="complex64")
        else:
            psi = cp.empty([len(big_psi), self.npsi, self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _DT(self, psi, big_psi):
            # pad to the probe size
            ff = cp.pad(big_psi, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)))
            # ff = big_psi.copy
            ff = cp.pad(ff,((0,0),(self.npsi//2,self.npsi//2),(self.npsi//2,self.npsi//2)))
            ff = cp.fft.ifft2(cp.fft.fft2(ff)/self.fker[cp.cuda.Device().id])
            v = cp.ones(2*self.npsi,dtype='float32')
            v[:self.npsi//2] = cp.sin(cp.linspace(0,1,self.npsi//2)*cp.pi/2)
            v[-self.npsi//2:] = cp.cos(cp.linspace(0,1,self.npsi//2)*cp.pi/2)
            v = cp.outer(v,v)
            ff *= v*2        
            psi[:] = self._adj_pad(ff)
        _DT(self, psi, big_psi)
        return psi
    
    def Dc(self, psi):
        """Forward propagator"""
 
        flg = chunking_flg(locals().values())        
        if flg:
            big_psi = np.empty([len(psi), self.npsi, self.npsi], dtype="complex64")
        else:
            big_psi = cp.empty([len(psi), self.npsi, self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _Dc(self, big_psi, psi):
    
            ff = psi.copy()
            ff = self._fwd_pad(ff)
            v = cp.ones(2*self.npsi,dtype='float32')
            v[:self.npsi//2] = cp.sin(cp.linspace(0,1,self.npsi//2)*cp.pi/2)
            v[-self.npsi//2:] = cp.cos(cp.linspace(0,1,self.npsi//2)*cp.pi/2)
            v = cp.outer(v,v)
            ff *= v*2        
            ff = cp.fft.ifft2(cp.fft.fft2(ff)*self.fkerc[cp.cuda.Device().id])
            ff = ff[:,self.npsi//2:-self.npsi//2,self.npsi//2:-self.npsi//2]
            big_psi[:] = ff

        _Dc(self, big_psi, psi)
        return big_psi

    def DcT(self, big_psi):
        """Adjoint propagator"""

        flg = chunking_flg(locals().values())        
        if flg:
            psi = np.empty([len(big_psi), self.npsi, self.npsi], dtype="complex64")
        else:
            psi = cp.empty([len(big_psi), self.npsi, self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _DcT(self, psi, big_psi):            
            # pad to the probe size
            ff = big_psi.copy()
            ff = cp.pad(ff,((0,0),(self.npsi//2,self.npsi//2),(self.npsi//2,self.npsi//2)))
            ff = cp.fft.ifft2(cp.fft.fft2(ff)/self.fkerc[cp.cuda.Device().id])
            v = cp.ones(2*self.npsi,dtype='float32')
            v[:self.npsi//2] = cp.sin(cp.linspace(0,1,self.npsi//2)*cp.pi/2)
            v[-self.npsi//2:] = cp.cos(cp.linspace(0,1,self.npsi//2)*cp.pi/2)
            v = cp.outer(v,v)
            ff *= v*2        
            psi[:] = self._adj_pad(ff)
            
        _DcT(self, psi, big_psi)
        return psi    


    def G(self,psi):
        # if self.lam==0:
            # return np.empty_like(psi)
        stencil = cp.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])
        res = cp.array(psi)#.copy()
        res = ndimage.convolve(res, stencil)
        return res.get()
    

            
    def gradientF(self, reused, d):
        big_psi = reused["big_psi"]
        flg = chunking_flg(locals().values())        
        if flg:
            gradF = np.empty([len(big_psi), self.npsi, self.npsi], dtype="complex64")
        else:
            gradF = cp.empty([len(big_psi), self.npsi, self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _gradientF(self, gradF, big_psi, d):
            td = d * (big_psi / (cp.abs(big_psi) + self.eps))
            gradF[:] = self.DT(2 * (big_psi - td))
            
        _gradientF(self, gradF, big_psi, d)
        reused['gradF'] = gradF            
    
    
    def gradient_psi(self, scode, gradF, psi, q):
        # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            gradpsi = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    gradpsi.append(cp.zeros([self.npsi, self.npsi], dtype="complex64"))
        else:
            gradpsi = cp.zeros([self.npsi, self.npsi], dtype="complex64")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_psi(self, gradpsi, scode, gradF, q):            
            gradpsi[:] += cp.sum(cp.conj(self.Dc(scode*q)) * gradF,axis=0)
        _gradient_psi(self, gradpsi, scode, gradF, q)
        if flg:
            for k in range(1,len(gradpsi)):
                gradpsi[0] += gradpsi[k]
            gradpsi = gradpsi[0].get()    
                
        gradpsi += 2*self.lam*self.G(self.G(psi))
        return gradpsi
    
    def gradient_q(self, scode, gradF, psi):
        flg = chunking_flg(locals().values())        
        if flg:
            gradq = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    gradq.append(cp.zeros([self.npsi, self.npsi], dtype="complex64"))
        else:
            gradq = cp.zeros([self.npsi, self.npsi], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                
        def _gradient_q(self, gradq, scode, gradF, psi):
            gradq[:] += cp.sum(cp.conj(scode) * self.DcT(cp.conj(psi)*gradF),axis=0)        
        _gradient_q(self, gradq, scode, gradF,  psi)

        if flg:
            for k in range(1,len(gradq)):
                gradq[0] += gradq[k]
            gradq = gradq[0]
            
        return gradq
   
    def gradient_r(self, ri, r, gradF, code, psi, q):
        
        flg = chunking_flg(locals().values())        
        if flg:
            gradr = np.empty([len(ri), 2], dtype="float32")
        else:
            gradr = cp.empty([len(ri), 2], dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)                               
        def _gradient_r(self, gradr, ri, r, gradF, code, psi, q):

            # frequencies
            xi1 = cp.fft.fftfreq(self.npatch).astype("float32")
            xi2, xi1 = cp.meshgrid(xi1, xi1)

            # multipliers in frequencies
            w = cp.exp(
                -2 * cp.pi * 1j * (xi2 * r[:, 1, None, None] + xi1 * r[:, 0, None, None])
            )

            # Gradient parts
            tmp = self.E(ri, code)
            tmp = cp.fft.fft2(tmp)
            
            dt1 = cp.fft.ifft2(w * xi1 * tmp)
            dt2 = cp.fft.ifft2(w * xi2 * tmp)
            dt1 = -2 * cp.pi * 1j * dt1[:, self.ex : self.npsi + self.ex, self.ex : self.npsi + self.ex]
            dt2 = -2 * cp.pi * 1j * dt2[:, self.ex : self.npsi + self.ex, self.ex : self.npsi + self.ex]

            # inner product with gradF

            dm1 = self.Dc(dt1*q)*psi
            dm2 = self.Dc(dt2*q)*psi

            gradr[:, 0] = redot(gradF, dm1, axis=(1, 2))
            gradr[:, 1] = redot(gradF, dm2, axis=(1, 2))
        
        _gradient_r(self, gradr, ri, r, gradF, code, psi, q)
        return gradr
    
    def gradients(self, vars, reused):
        (q, code, psi, ri, r) = (vars["q"], vars["code"], vars["psi"], vars["ri"], vars["r"])
        (gradF, scode) = (reused["gradF"], reused["scode"])

        dpsi = self.gradient_psi(scode, gradF, psi, q)
        dprb = self.gradient_q(scode, gradF, psi)
        dr = self.gradient_r(ri, r, gradF, code, psi, q)
        grads = {"psi": dpsi, "q": dprb, "r": dr}
        return grads
    
    def minF(self, big_psi, d):
        flg = chunking_flg(locals().values())        
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)          
        def _minF(self, res, big_psi, d):            
            res[:]+=np.linalg.norm(cp.abs(big_psi) - d) ** 2
        _minF(self, res, big_psi, d)
        if flg:
            for k in range(1,len(res)):
                res[0] += res[k]
            res = res[0]    
        res = res[0]    
        return res

    def calc_reused(self, reused, vars):
        """Calculate variables reused during calculations"""
        (q, code, psi, ri, r) = (vars["q"], vars['code'], vars["psi"], vars["ri"], vars["r"])
        flg = chunking_flg(vars.values())   
        if flg:
            scode = np.empty([len(ri), self.npsi, self.npsi], dtype="complex64")
            big_psi = np.empty([len(ri), self.n, self.n], dtype="complex64")
        else:
            scode = cp.empty([len(ri), self.npsi, self.npsi], dtype="complex64")
            big_psi = cp.empty([len(ri), self.n, self.n], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_reused1(self, scode, ri, r, code):
            scode[:] = self.S(ri, r, code)
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_reused2(self, big_psi, scode, psi, q):
            big_psi[:] = self.D(self.Dc(scode*q)*psi)            

        _calc_reused1(self, scode, ri, r, code)        
        _calc_reused2(self, big_psi, scode, psi, q)            
        
        reused["scode"] = scode
        reused["big_psi"] = big_psi
                
    
    def hessian_F(self, big_psi, dbig_psi1, dbig_psi2, d):    
        flg = chunking_flg(locals().values())        
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="complex64"))
        else:
            res = cp.zeros(1, dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _hessian_F(self, res, big_psi, dbig_psi1, dbig_psi2, d):
            l0 = big_psi / (cp.abs(big_psi) + self.eps)
            d0 = d / (cp.abs(big_psi) + self.eps)
            v1 = cp.sum((1 - d0) * reprod(dbig_psi1, dbig_psi2))
            v2 = cp.sum(d0 * reprod(l0, dbig_psi1) * reprod(l0, dbig_psi2))
            res[:] = 2 * (v1 + v2)
            return res
        _hessian_F(self, res, big_psi, dbig_psi1, dbig_psi2, d)
        if flg:
            for k in range(1,len(res)):
                res[0] += res[k]
        res = res[0]    
        return res

    def redot_batch(self,x,y):
        flg = chunking_flg(locals().values())        
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros(1, dtype="float32"))
        else:
            res = cp.zeros(1, dtype="float32")
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _redot(self, res, x,y):
            res[:] += redot(x, y)
            return res
        _redot(self, res, x,y)
        if flg:
            for k in range(1,len(res)):
                res[0] += res[k]
            res = res[0]    
        res = res[0]
        return res
    
    def calc_beta(self, vars, grads, etas, reused, d):
        (q, code, psi, ri, r) = (vars["q"], vars["code"], vars["psi"], vars["ri"], vars["r"])
        (scode, big_psi, gradF) = (reused["scode"], reused["big_psi"], reused["gradF"])
        (dpsi1, dq1, dr1) = (grads["psi"], grads["q"], grads["r"])
        (dpsi2, dq2, dr2) = (etas["psi"], etas["q"], etas["r"])        

        flg = (chunking_flg(vars.values()) 
               or chunking_flg(reused.values()) 
               or chunking_flg(grads.values()) 
               or chunking_flg(etas.values()))                 
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros([2], dtype="float32"))
        else:
            res = cp.zeros([2], dtype="float32")

        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_beta(self, res, ri, r, scode, big_psi, gradF, dr1, dr2, d, q, dq1, dq2, code, psi,dpsi1, dpsi2):            
            # note scaling with rho
            dpsi1 = dpsi1 * self.rho[0] ** 2
            dq1 = dq1 * self.rho[1] ** 2
            dr1 = dr1 * self.rho[2] ** 2
            # frequencies
            xi1 = cp.fft.fftfreq(self.npatch).astype("float32")
            [xi2, xi1] = cp.meshgrid(xi1, xi1)

            # multipliers in frequencies
            
            dr1 = dr1[:, :, cp.newaxis, cp.newaxis]
            dr2 = dr2[:, :, cp.newaxis, cp.newaxis]
            w = cp.exp(
                -2 * cp.pi * 1j * (xi2 * r[:, 1, None, None] + xi1 * r[:, 0, None, None])
            )
            w1 = xi1 * dr1[:, 0] + xi2 * dr1[:, 1]
            w2 = xi1 * dr2[:, 0] + xi2 * dr2[:, 1]
            w12 = (
                xi1**2 * dr1[:, 0] * dr2[:, 0]
                + xi1 * xi2 * (dr1[:, 0] * dr2[:, 1] + dr1[:, 1] * dr2[:, 0])
                + xi2**2 * dr1[:, 1] * dr2[:, 1]
            )
            w22 = (
                xi1**2 * dr2[:, 0] ** 2
                + 2 * xi1 * xi2 * (dr2[:, 0] * dr2[:, 1])
                + xi2**2 * dr2[:, 1] ** 2
            )

            tmp = self.E(ri, code)
            tmp = cp.fft.fft2(tmp)
            dt1 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)[:, self.ex : self.npsi + self.ex, self.ex : self.npsi + self.ex]
            dt2 = -2 * cp.pi * 1j * cp.fft.ifft2(w * w2 * tmp)[:, self.ex : self.npsi + self.ex, self.ex : self.npsi + self.ex]
            d2t1 = -4 * cp.pi**2 * cp.fft.ifft2(w * w12 * tmp)[:, self.ex : self.npsi + self.ex, self.ex : self.npsi + self.ex]
            d2t2 = -4 * cp.pi**2 * cp.fft.ifft2(w * w22 * tmp)[:, self.ex : self.npsi + self.ex, self.ex : self.npsi + self.ex]

            # DM,D2M terms
            
            d2m1 = q * d2t1
            d2m1 += dq1 * dt2 + dq2 * dt1

            d2m2 = q * d2t2
            d2m2 += dq2 * dt2 + dq2 * dt2

            dm1 = dq1 * scode + q * dt1
            dm2 = dq2 * scode + q * dt2

            dn1 = self.Dc(dm1)*psi + self.Dc(q*scode)*dpsi1
            dn2 = self.Dc(dm2)*psi + self.Dc(q*scode)*dpsi2
        
            d2n1 = self.Dc(d2m1)*psi + self.Dc(dm1)*dpsi2+ self.Dc(dm2)*dpsi1
            d2n2 = self.Dc(d2m2)*psi + self.Dc(dm2)*dpsi2+ self.Dc(dm2)*dpsi2
            # top and bottom parts
            Ddn1 = self.D(dn1)
            Ddn2 = self.D(dn2)
            top = redot(gradF, d2n1) + self.hessian_F(big_psi, Ddn1, Ddn2, d)
            bottom = redot(gradF, d2n2) + self.hessian_F(big_psi, Ddn2, Ddn2, d)
            res[0] += top
            res[1] += bottom
            
        _calc_beta(self, res, ri, r, scode, big_psi, gradF, dr1, dr2, d, q, dq1, dq2, code, psi,dpsi1, dpsi2)           
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]
            res = res[0]      
        
                
        if self.lam>0:
            gpsi1 = self.G(dpsi1)
            gpsi2 = self.G(dpsi2)
            res[0] += 2*self.lam*self.redot_batch(gpsi1,gpsi2)
            res[1] += 2*self.lam*self.redot_batch(gpsi2,gpsi2)

        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom
    

    def calc_alpha(self, vars, grads, etas, reused, d):
        (q, code, psi, ri, r) = (vars["q"], vars["code"], vars["psi"], vars["ri"], vars["r"])
        (dpsi1, dq1, dr1) = (grads["psi"], grads["q"], grads["r"])
        (dpsi2, dq2, dr2) = (etas["psi"], etas["q"], etas["r"])
        (scode, big_psi, gradF) = (reused["scode"], reused["big_psi"], reused["gradF"])
        
        flg = (chunking_flg(vars.values()) 
               or chunking_flg(reused.values()) 
               or chunking_flg(grads.values()) 
               or chunking_flg(etas.values()))                 
        if flg:
            res = []
            for igpu in range(self.ngpus):
                with cp.cuda.Device(igpu):
                    res.append(cp.zeros([2], dtype="float32"))
        else:
            res = cp.zeros([2], dtype="float32")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _calc_alpha(self, res, ri, r, scode, big_psi, gradF, dr1, dr2, d, q, dq2, code, psi, dpsi2):            
            # top part
            top = -redot(dr1, dr2)

            # frequencies
            xi1 = cp.fft.fftfreq(self.npatch).astype("float32")
            [xi2, xi1] = cp.meshgrid(xi1, xi1)

            # multipliers in frequencies
            dr = dr2[:, :, cp.newaxis, cp.newaxis]
            w = cp.exp(
                -2 * cp.pi * 1j * (xi2 * r[:, 1, None, None] + xi1 * r[:, 0, None, None])
            )
            w1 = xi1 * dr[:, 0] + xi2 * dr[:, 1]
            w2 = (
                xi1**2 * dr[:, 0] ** 2
                + 2 * xi1 * xi2 * (dr[:, 0] * dr[:, 1])
                + xi2**2 * dr[:, 1] ** 2
            )

            # DT,D2T terms, and scode
            
            tmp = self.E(ri, code)
            tmp = cp.fft.fft2(tmp)
            dt = -2 * cp.pi * 1j * cp.fft.ifft2(w * w1 * tmp)[:, self.ex : self.npsi + self.ex, self.ex : self.npsi + self.ex]
            d2t = -4 * cp.pi**2 * cp.fft.ifft2(w * w2 * tmp)[:, self.ex : self.npsi + self.ex, self.ex : self.npsi + self.ex]
            d2m2 = q * d2t
            d2m2 += dq2 * dt + dq2 * dt
            
            dm2 = dq2 * scode + q * dt
            
            dn2 = self.Dc(dm2)*psi + self.Dc(q*scode)*dpsi2
            d2n2 = self.Dc(d2m2)*psi + self.Dc(dm2)*dpsi2+ self.Dc(dm2)*dpsi2
            # top and bottom parts
            
            Ddn2 = self.D(dn2)
            bottom = redot(gradF, d2n2) 
            bottom += self.hessian_F(big_psi, Ddn2, Ddn2, d)
            res[0] += top
            res[1] += bottom
            

        _calc_alpha(self, res, ri, r, scode, big_psi, gradF, dr1, dr2, d, q, dq2, code, psi, dpsi2)
        if flg:
            for k in range(1,len(res)):
                res[0][0] += res[k][0]
                res[0][1] += res[k][1]                
            res = res[0]
        res[0] += -redot(dpsi1, dpsi2) - redot(dq1, dq2)
      
        gpsi2 = self.G(dpsi2)
        res[1] += 2*self.lam*redot(gpsi2,gpsi2)

        top = cp.float32((res[0]).get()) 
        bottom = cp.float32((res[1]).get()) 
        return top/bottom,top,bottom
    
    def fwd(self,ri,r,code, psi,q):
                
         # memory for result
        flg = chunking_flg(locals().values())        
        if flg:
            res = np.empty([len(ri), self.n, self.n], dtype="complex64")
        else:
            res = cp.empty([len(ri), self.n, self.n], dtype="complex64")
        
        @gpu_batch(self.nchunk, self.ngpus, axis_out=0, axis_inp=0)
        def _fwd(self,res,ri,r,code,psi,q):
            res[:] = self.D(self.Dc(self.S(ri,r,code) * q)*psi)
                     
        _fwd(self,res,ri,r,code,psi,q)
        return res

    def plot_debug(self, vars, etas, top, bottom, alpha, d, i):
        """Check the minimization functional behaviour"""
        if i % self.vis_step == 0 and self.vis_step != -1 and self.show:
            (q, code, psi, ri, r) = (vars["q"], vars["code"], vars["psi"], vars["ri"], vars["r"])
            (dq2, dpsi2, dr2) = (etas["q"], etas["psi"], etas["r"])
                        
                        
            npp = 3
            errt = np.zeros(npp * 2)
            errt2 = np.zeros(npp * 2)
            for k in range(0, npp * 2):
                psit = psi + (alpha * k / (npp - 1)) * dpsi2
                qt = q + (alpha * k / (npp - 1)) * dq2
                rt = r + (alpha * k / (npp - 1)) * dr2
                tmp=self.fwd(ri,rt,code,psit,qt)
                errt[k] = self.minF(tmp, d)
                if self.lam>0:
                    errt[k]+= self.lam*np.linalg.norm(self.G(psit))**2 
                
            t = alpha * (np.arange(2 * npp)) / (npp - 1)
            tmp = self.fwd(ri,r,code,psi,q)
            errt2 = self.minF(tmp, d).get() 
            if self.lam>0:
                errt2+= self.lam*np.linalg.norm(self.G(psi))**2 
            errt2 = errt2 - top * t + 0.5 * bottom * t**2

            xaxis = alpha * np.arange(2 * npp) / (npp - 1)

            plt.plot(xaxis,errt,".",label="real")
            plt.plot(xaxis,errt2,".",label="approx")
            plt.legend()
            plt.grid()
            plt.show()

    def vis_debug(self, vars, i):
        """Visualization and data saving"""
        if i % self.vis_step == 0 and self.vis_step != -1:
            (q, psi) = (vars["q"], vars["psi"])
            mshow_polar(psi,self.show)
            mshow_polar(q,self.show)
            
            write_tiff(np.angle(psi),f'{self.path_out}/rec_psi_angle/{i:04}')
            write_tiff(np.abs(psi),f'{self.path_out}/rec_psi_abs/{i:04}')
            write_tiff(np.angle(q),f'{self.path_out}/rec_prb_angle/{i:04}')
            write_tiff(np.abs(q),f'{self.path_out}/rec_prb_abs/{i:04}')
            mplot_positions(vars["r"],self.show)
            
    def error_debug(self,vars, reused, d, i):
        """Visualization and data saving"""
        psi = vars['psi']
        if i % self.err_step == 0 and self.err_step != -1:
            err = self.minF(reused["big_psi"], d) + self.lam*np.linalg.norm(self.G(psi))**2 
            print(f"{i}) {err=:1.5e}", flush=True)
            vars["table"].loc[len(vars["table"])] = [i, err.get(), time.time()]
            name = f'{self.path_out}/conv.csv'
            os.makedirs(os.path.dirname(name), exist_ok=True)
            vars["table"].to_csv(name, index=False)
    