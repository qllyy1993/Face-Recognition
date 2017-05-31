clc,clear
npersons=40;%number of person is 40
global imgrow;
global imgcol;
%global edit2
imgrow=112;% image row pixel
imgcol=92;% image column pixel

%set(edit2,'string','Read training data...')
%drawnow
disp('Read training data......')
disp('.................................................')
f_matrix=ReadFace(npersons,0);%read training data
nfaces=size(f_matrix,1);%number of sample faces

%set(edit2,'string','Training data PCA feature extraction ...')
disp('Training data feature extraction...')
disp('.................................................')
%drawnow
mA=mean(f_matrix);
k=20;%reducing dimensionality to k
[pcaface,V]=fastPCA(f_matrix,k,mA);%feature extraction using PCA

%set(edit2,'string','Training feature data normalization...')
disp('Training feature data normalizaition......')
disp('.................................................')
%drawnow
lowvec=min(pcaface);
upvec=max(pcaface);
scaledface = scaling( pcaface,lowvec,upvec);
disp('SVMsamples training......')
disp('.................................................')
%set(edit2,'string','SVM sample training......')
%drawnow
gamma=0.0078;%best parameters
c=128;
multiSVMstruct=multiSVMtrain( scaledface,npersons,gamma,c);
save('recognize.mat','multiSVMstruct','npersons','k','mA','V','lowvec','upvec');
disp('read test data......')
disp('.................................................')
%set(edit2,'string','Read test data......')
%drawnow
[testface,realclass]=ReadFace(npersons,1);

%set(edit2,'string','Test data reducing dimension......')
%drawnow
disp('Test data feature dimension reduction......')
disp('.................................................')
m=size(testface,1);
for i=1:m
    testface(i,:)=testface(i,:)-mA;
end
pcatestface=testface*V;

%set(edit2,'string','Test feature data normalization......')
%drawnow
scaledtestface = scaling( pcatestface,lowvec,upvec);
disp('SVM sample classification...')
disp('.................................................')
%set(edit2,'string','SVM sample classify......')
%drawnow
class= multiSVM(scaledtestface,multiSVMstruct,npersons);
%set(edit2,'string','Accuracy:')
accuracy=sum(class==realclass)/length(class);
msgbox(['Recognition Accuracy:',num2str(accuracy*100),'%'])