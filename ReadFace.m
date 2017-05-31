function [f_matrix,realclass] = ReadFace(npersons,flag)
%get the data from ORL database to matlab
%%Input:
%nperson---total number of person of samples
%num_train---the number of images for everyone
%num_test---the number of images for everyone
%imgrow --- the row pixel of image
%imgcol --- the column pixel of image
%imgrow=112; imgcol=92;
global imgrow;
global imgcol;
realclass=zeros(npersons*5,1);
f_matrix=zeros(npersons*5,imgrow*imgcol);
for i=1:npersons
    facepath='orl_faces/s';
    facepath=strcat(facepath,num2str(i));
    facepath=strcat(facepath,'/');
    cachepath=facepath;
    for j=1:5
        facepath=cachepath;
        if flag==0 %input the training samples
            facepath=strcat(facepath,'0'+j);
        else %input the testing samples
            facepath=strcat(facepath,num2str(5+j));
            realclass((i-1)*5+j)=i;
        end
        facepath=strcat(facepath,'.pgm');
        img1=imread(facepath);
       % img2=gray_comp(img1);
%         figure
%         imshow(img)
        f_matrix((i-1)*5+j,:)=img1(:)';
    end
end
end