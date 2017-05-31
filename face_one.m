clc,clear
global edit2
npersons=40;%number of person is 40
global imgrow;
global imgcol;
imgrow=112;
imgcol=92;

set(edit2,'string','Read training data...')
drawnow
disp('Read training data......')
disp('.................................................')
f_matrix=ReadFace(npersons,0);%read training data

set(edit2,'string','training data feature extraction......')
drawnow
disp('training data feature extraction...')
disp('.................................................')
k=20;%reducing dimension to 20
m=size(f_matrix,1);
mA=mean(f_matrix);%means of samples
Z=(f_matrix-repmat(mA,m,1));
T=Z*Z';
[V,D]=eigs(T,k);%calculating the k principal eigenvalue and eigenvector
V=Z'*V;%eigenvetor of covariance matrix 
for i=1:k%unitization
    l=norm(V(:,i));
    V(:,i)=V(:,i)/l;
end
pcaface=Z*V;%linear transform, reduced to dimension

approx=mA;
for i=1:k
    approx=approx+pcaface(55,i)*V(:,i)';%the 1st person to reconstructing face
end
disp('Reconstructing face')
figure
B=reshape(approx',112,92);
imshow(B,[])

% disp('Showing the principal component face...')
visualize(V)%Showing the principal component face
disp('.................................................')

% disp('Press any key to continue...')
%pause
set(edit2,'string','training data normalization......')
drawnow
disp('training feature data normalizaition......')
disp('.................................................')
lowvec=min(pcaface);
upvec=max(pcaface);
upnew=1;
lownew=-1;
[m,n]=size(pcaface);
scaledface=zeros(m,n);
for i=1:m
    scaledface(i,:)=lownew+(pcaface(i,:)-lowvec)./(upvec-lowvec)*(upnew-lownew);
end
set(edit2,'string','SVM samples training......')
drawnow
disp('SVMsamples training......')
disp('.................................................')
gamma=0.0078;
c=128;% different punishment
for i=1:npersons-1
    for j=i+1:npersons
        X=[scaledface(5*(i-1)+1:5*i,:);scaledface(5*(j-1)+1:5*j,:)];
        Y=[ones(5,1);zeros(5,1)];
        multiSVMstruct{i}{j}=svmtrain(X,Y,'Kernel_Function',@(X,Y) kfun_rbf(X,Y,gamma),'boxconstraint',c);%,'rbf','RBF_Sigma',gamma,'boxconstraint',c);
    end
end

save('recognize.mat','multiSVMstruct','npersons','k','mA','V','lowvec','upvec');
set(edit2,'string','read test data......')
drawnow
disp('read test data......')
disp('.................................................')
testpersons=40;
[testface,realclass]=ReadFace(testpersons,1);
set(edit2,'string','Test data feature dimension reduction......')
drawnow
disp('Test data feature dimension reduction......')
disp('.................................................')
m=size(testface,1);
Z=testface-repmat(mA,m,1);
pcatestface=Z*V;

% approx=mA;
% for i=1:k
%     approx=approx+pcatestface(1,i)*V(:,i)';
% end
% disp('Face reconstruction')
% figure
% B=reshape(approx',112,92);
% imshow(B,[])
%set(edit2,'string','Test data normalization......')
%drawnow
disp('Test data normalization¯¯...')
disp('.................................................')
[m,n]=size(pcatestface);
scaledtestface=zeros(m,n);
for i=1:m
    scaledtestface(i,:)=lownew+(pcatestface(i,:)-lowvec)./(upvec-lowvec)*(upnew-lownew);
end
set(edit2,'string','SVM samples classification......')
drawnow
disp('SVM sample classification...')
disp('.................................................')
voting=zeros(m,npersons);
for i=1:npersons-1
    for j=i+1:npersons
        class=svmclassify(multiSVMstruct{i}{j},scaledtestface);
        voting(:,i)=voting(:,i)+(class==1);
        voting(:,j)=voting(:,j)+(class==0);
    end
end
%  disp(voting)
[~,class]=max(voting,[],2);
set(edit2,'string','Test finishing!')
accuracy=sum(class==realclass)/length(class);
display(['Accuracy:',num2str(accuracy)])
msgbox(['Accuracy:',num2str(accuracy*100),'%.'])