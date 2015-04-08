
clear;
clc;

global a ;
%myimage=imread('C:\Users\Anand Namdev\Documents\Visual Studio 2013\Projects\minor\image data set\cameraman.tif'); 
myimage=a;
global ori;
ori=myimage;
%figure,imshow(myimage),title('original image');

[row col dim]=size(myimage);  

if dim==3
    
     k3=parallel.gpu.CUDAKernel('C:\Users\Anand Namdev\Documents\Visual Studio 2013\Projects\minor\kernel3.ptx','C:\Users\Anand Namdev\Documents\Visual Studio 2013\Projects\minor\kernel3.cu');
     blockSize=32;
     k3.ThreadBlockSize = [blockSize,1,3];
     k3.GridSize=[ceil(row/blockSize),col];
     
    
     myimage_double=double(myimage); 
     
     tic;
     myimage_gpu=gpuArray(myimage_double);
     out1=gpuArray.zeros(size(myimage(:,:,1)),'double');
     out2=gpuArray.zeros(size(myimage(:,:,1)),'double');
     out3=gpuArray.zeros(size(myimage(:,:,1)),'double');
     toc;
     
     tic;
     out1=feval(k3,out1,myimage_gpu,row,col,0); 
     out2=feval(k3,out2,myimage_gpu,row,col,1); 
     out3=feval(k3,out3,myimage_gpu,row,col,2); 
     toc;
     tic;
     result_convertR_cpu=gather(out1);
     result_convertG_cpu=gather(out2);
     result_convertB_cpu=gather(out3);
     toc;
     
    
     
   
     R=result_convertR_cpu;
     G=result_convertG_cpu;
     B=result_convertB_cpu;
 
    R=double(R);
    G=double(G);
    B=double(B);
    gpuDevice(1);
    
     k=parallel.gpu.CUDAKernel('C:\Users\Anand Namdev\Documents\Visual Studio 2013\Projects\minor\kernel.ptx','C:\Users\Anand Namdev\Documents\Visual Studio 2013\Projects\minor\kernel.cu');
    blockSize=32;
    k.ThreadBlockSize = [blockSize,1,3];
    k.GridSize=[ceil(row/blockSize),col];
    
    
    
    
    
    tic;  
    R_gpu=gpuArray(R);
    G_gpu=gpuArray(G);
    B_gpu=gpuArray(B);
    gray=gpuArray.zeros(size(R),'double');
     
    gray=feval(k,gray,R_gpu,G_gpu,B_gpu,row,col);
    result_CPU=gather(gray);
    toc;  
        
    
else
    result_CPU=double(myimage);
end

%figure(),imshow(uint8(result_CPU)),title('grayscale');
global gs;
gs=uint8(result_CPU);
k_sobel=parallel.gpu.CUDAKernel('C:\Users\Anand Namdev\Documents\Visual Studio 2013\Projects\minor\kernel2.ptx','C:\Users\Anand Namdev\Documents\Visual Studio 2013\Projects\minor\kernel2.cu');
blockSize=32;
k_sobel.ThreadBlockSize = [blockSize,1,3];
k_sobel.GridSize=[ceil(row/blockSize),col];


grayimage_GPU=result_CPU;


tic;    
grayimage_GPU=gpuArray(grayimage_GPU);
gradient=gpuArray.zeros(size(myimage(:,:,1)),'double');

gradient=feval(k_sobel,gradient,grayimage_GPU,row,col);
result2_CPU=gather(gradient);
toc;    





result2_CPU=uint8(result2_CPU);
global grad;
grad=result2_CPU;

%figure(),imshow(result2_CPU),title('Gradient');


final=im2bw(result2_CPU);

%figure,imshow(final),title('final');
thin=bwmorph(final,'skel');
global fn;
fn=thin;
%figure,imshow(thin),title('thin');


so=edge(result_CPU,'sobel');
global soo;
soo=so;
%figure,imshow(so),title('by sobel function');


