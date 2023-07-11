%function [reco,vect]=lda(featno,trns,feature,fsv)
% This function classifies the phonemes using linear discriminant analysis.
% First it open the training file containing the features.
% featno is the number of features. 
% trns=1 for AWP features; trns=2 for mfcc features.
% feature is the feature vector for classification.
% fsv =1 for fricatives; fsv=2 for stops; fsv=3 for vowels.

clear all
clc
fun='diagquadratic';
per=0.70;
fid0=fopen('E:\research\sifer.txt','r');
[in0 count0]=fscanf(fid0,'%f');
class0=count0/52;
P0=reshape(in0,52,class0);
class0_tr=floor(class0*per);
class0_ts=class0-class0_tr;
P0_tr=P0(:,1:class0_tr);
P0_ts=P0(:,class0_tr+1:end);

fid1=fopen('E:\research\yow.txt','r');
[in1 count1]=fscanf(fid1,'%f');
class1=count1/52;
P1=reshape(in1,52,class1);
class1_tr=floor(class1*per);
class1_ts=class1-class1_tr;
P1_tr=P1(:,1:class1_tr);
P1_ts=P1(:,class1_tr+1:end);

fid2=fopen('E:\research\dwa.txt','r');
[in2 count2]=fscanf(fid2,'%f');
class2=count2/52;
P2=reshape(in2,52,class2);
class2_tr=floor(class2*per);
class2_ts=class2-class2_tr;
P2_tr=P2(:,1:class2_tr);
P2_ts=P2(:,class2_tr+1:end);

fid3=fopen('E:\research\dray.txt','r');
[in3 count3]=fscanf(fid3,'%f');
class3=count3/52;
P3=reshape(in3,52,class3);
class3_tr=floor(class3*per);
class3_ts=class3-class3_tr;
P3_tr=P3(:,1:class3_tr);
P3_ts=P3(:,class3_tr+1:end);

fid4=fopen('E:\research\celour.txt','r');
[in4 count4]=fscanf(fid4,'%f');
class4=count4/52;
P4=reshape(in4,52,class4);
class4_tr=floor(class4*per);
class4_ts=class4-class4_tr;
P4_tr=P4(:,1:class4_tr);
P4_ts=P4(:,class4_tr+1:end);

fid5=fopen('E:\research\pinza.txt','r');
[in5 count5]=fscanf(fid5,'%f');
class5=count5/52;
P5=reshape(in5,52,class5);
class5_tr=floor(class5*per);
class5_ts=class5-class5_tr;
P5_tr=P5(:,1:class5_tr);
P5_ts=P5(:,class5_tr+1:end);

fid6=fopen('E:\research\shpeg.txt','r');
[in6 count6]=fscanf(fid6,'%f');
class6=count6/52;
P6=reshape(in6,52,class6);
class6_tr=floor(class6*per);
class6_ts=class6-class6_tr;
P6_tr=P6(:,1:class6_tr);
P6_ts=P6(:,class6_tr+1:end);

fid7=fopen('E:\research\ova.txt','r');
[in7 count7]=fscanf(fid7,'%f');
class7=count7/52;
P7=reshape(in7,52,class7);
class7_tr=floor(class7*per);
class7_ts=class7-class7_tr;
P7_tr=P7(:,1:class7_tr);
P7_ts=P7(:,class7_tr+1:end);

fid8=fopen('E:\research\ata.txt','r');
[in8 count8]=fscanf(fid8,'%f');
class8=count8/52;
P8=reshape(in8,52,class8);
class8_tr=floor(class8*per);
class8_ts=class8-class8_tr;
P8_tr=P8(:,1:class8_tr);
P8_ts=P8(:,class8_tr+1:end);

fid9=fopen('E:\research\naha.txt','r');
[in9 count9]=fscanf(fid9,'%f');
class9=count9/52;
P9=reshape(in9,52,class9);
class9_tr=floor(class9*per);
class9_ts=class9-class9_tr;
P9_tr=P9(:,1:class9_tr);
P9_ts=P9(:,class9_tr+1:end);

fid10=fopen('E:\research\las.txt','r');
[in10 count10]=fscanf(fid10,'%f');
class10=count10/52;
P10=reshape(in10,52,class10);
class10_tr=floor(class10*per);
class10_ts=class10-class10_tr;
P10_tr=P10(:,1:class10_tr);
P10_ts=P10(:,class10_tr+1:end);

g0=ones(1,class0_tr);
g1=2*ones(1,class1_tr);
g2=3*ones(1,class2_tr);
g3=4*ones(1,class3_tr);
g4=5*ones(1,class4_tr);
g5=6*ones(1,class5_tr);
g6=7*ones(1,class6_tr);
g7=8*ones(1,class7_tr);
g8=9*ones(1,class8_tr);
g9=10*ones(1,class9_tr);
g10=11*ones(1,class10_tr);
g=cat(1,g0',g1',g2',g3',g4',g5',g6',g7',g8',g9',g10');
train=cat(1,P0_tr',P1_tr',P2_tr',P3_tr',P4_tr',P5_tr',P6_tr',P7_tr',P8_tr',P9_tr',P10_tr');
pp=0;pt=0;pk=0;

class_tr=[class0_tr,class1_tr,class2_tr,class3_tr,class4_tr,class5_tr,class6_tr, ...
    class7_tr,class8_tr,class9_tr,class10_tr];
fin=0;
for j=1:11
    fin=class_tr(j)+fin;
    ini=fin-class_tr(j)+1;
    dat=train(ini:fin,:);
    %qq=kmeans(train,10);
    %[pn,meanp,stdp] = prestd(train');
    %[ptrans,transMat] = prepca(pn,0.02)
    sifer=0;yow=0;dwa=0;dray=0;celour=0;pinza=0;shpeg=0;ova=0;ata=0;naha=0;las=0;
    [classp]=classify(dat,train,g,fun);
    for i=1:length(classp)
        if classp(i) == 1 
            reco(i)=1;
            sifer=sifer+1;
        elseif classp(i) ==2
            reco(i)=2;
            yow=yow+1;
        elseif classp(i) ==3
            reco(i)=3;
            dwa=dwa+1;
        elseif classp(i) ==4
            reco(i)=4;
            dray=dray+1;
        elseif classp(i) ==5
            reco(i)=5; 
            celour=celour+1;
       elseif classp(i) ==6
            reco(i)=6; 
            pinza=pinza+1;
       elseif classp(i) ==7
            reco(i)=7; 
            shpeg=shpeg+1;
       elseif classp(i) ==8
            reco(i)=8; 
            ova=ova+1;
       elseif classp(i) ==9
            reco(i)=8; 
            ata=ata+1;
        elseif classp(i) ==10
            reco(i)=9; 
            naha=naha+1;
      
        else
            reco(i)=11;
            las=las+1;
        end
    end
    reco;
    con(j,:)=[sifer yow dwa dray celour pinza shpeg ova ata naha las];
    per_corr(j)=con(j,j)/class_tr(j);
end
con
per_corr




pp=0;pt=0;pk=0;
test=cat(1,P0_ts',P1_ts',P2_ts',P3_ts',P4_ts',P5_ts',P6_ts',P7_ts',P8_ts',P9_ts',P10_ts');

class_ts=[class0_ts,class1_ts,class2_ts,class3_ts,class4_ts,class5_ts,class6_ts,class7_ts,class8_ts,class9_ts,class10_ts];
fin=0;
for j=1:11
    fin=class_ts(j)+fin;
    ini=fin-class_ts(j)+1;
    dat=test(ini:fin,:);
    %qq=kmeans(train,10);
    %[pn,meanp,stdp] = prestd(train');
    %[ptrans,transMat] = prepca(pn,0.02)
    sifer=0;yow=0;dwa=0;dray=0;celour=0;pinza=0;shpeg=0;ova=0;ata=0;naha=0;las=0;
    [classp]=classify(dat,train,g,fun);
    for i=1:length(classp)
        if classp(i) == 1 
            reco(i)=1;
            sifer=sifer+1;
        elseif classp(i) ==2
            reco(i)=2;
            yow=yow+1;
        elseif classp(i) ==3
            reco(i)=3;
            dwa=dwa+1;
        elseif classp(i) ==4
            reco(i)=4;
            dray=dray+1;
        elseif classp(i) ==5
            reco(i)=5; 
            celour=celour+1;
          elseif classp(i) ==6
            reco(i)=6; 
            pinza=pinza+1;
       elseif classp(i) ==7
            reco(i)=7; 
            shpeg=shpeg+1;
       
       elseif classp(i) ==8
            reco(i)=8; 
            ova=ova+1;
       elseif classp(i) ==9
            reco(i)=9; 
            ata=ata+1;
      
      elseif classp(i) ==10
            reco(i)=10; 
            naha=naha+1;
      
        else
            reco(i)=11;
            las=las+1;
     end
    end
    reco;
    con(j,:)=[sifer yow dwa dray celour pinza shpeg ova ata naha las];
    per_corr(j)=con(j,j)/class_ts(j);
end
con
per_corr

