% serial code to detct the edges of an image 
% project under category parallel image processing using CUDA C .



% clear the history using clear and clc command 
clc;
clear;

%Firstly read the image
%Give the path of the image whose edge you want to detect
myimage=imread('C:\Users\Anand Namdev\Documents\Visual Studio 2013\Projects\minor\image data set\1.png');
figure,imshow(myimage),title('original image');
% this will give the row,column and dimension of the selected image
[row col d]=size(myimage);
tic;  %begin timer
if d== 3   % a rgb image will have 3 components R,G,B .
    for i=1:row
        for j=1:col
            red(i,j)=myimage(i,j,1);    %returns red component of rgb image
            green(i,j)=myimage(i,j,2);  %returns green component of rgb image
            blue(i,j)=myimage(i,j,3);   %returns blue component of rgb image
            %for each pixel find equivalent gray intensity
            gray(i,j)=red(i,j)*0.299 + green(i,j)*0.587 + blue(i,j)*0.114;
        end
    end
end
toc;  %end timer



figure,imshow(gray),title('grayscale image');

%type cast uint8 to double
grayDouble=double(gray);
tic;  %begin timer

%Applying sobel's mask

for i=1:row-2
    for j=1:col-2
        %calculating gradient in x direction 
        %     | -1 0 1 |
        % H=  | -2 0 2 |
        %     | -1 0 1 |
        
        Gx(i,j)=double(((2*grayDouble(i+2,j+1)+grayDouble(i+2,j)+grayDouble(i+2,j+2))-(2*grayDouble(i,j+1)+grayDouble(i,j)+grayDouble(i,j+2))));
        
        
        %calculating gradient in y direction 
        %     | -1 -2 -1|
        % V=  | 0  0  0 |
        %     | 1  2  1 |
            
        Gy(i,j)=double(((2*grayDouble(i+1,j+2)+grayDouble(i,j+2)+grayDouble(i+2,j+2))-(2*grayDouble(i+1,j)+grayDouble(i,j)+grayDouble(i+2,j))));
        
        sobel_grad(i,j)=sqrt(Gx(i,j).^2+Gy(i,j).^2);
    end
end
toc;   %end timer

figure,imshow(Gx/255),title('horizontal');
figure,imshow(Gy/255),title('vertical');
figure,imshow(sobel_grad/255),title('gradient sobel ');

% by imwrite we create an image of name test_me which is sobel_gradient 
imwrite(sobel_grad/255,'test_me.tif');  

final=im2bw(sobel_grad/255);
figure,imshow(final),title('final');
so=edge(gray,'sobel');   % sobel's inbuilt function 
figure,imshow(so),title('sobel edge');




