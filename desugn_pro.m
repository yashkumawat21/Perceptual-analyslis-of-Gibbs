clc;
clear;
close all;
T = readtable('user13.txt');

F1 = [T{:,"N1"}];
F2 = [T{:,"N2"}];
flag = [T{:,"Flag"}];

seq=(flag==1);
seq1=~seq;
flag(seq1)=-1;

Data=[F1 F2 flag];

Data_per=Data(seq,:);
Data_nper=Data(seq1,:);

clear f1;
clear f2;
clear flag;
%clear last;
clear F1 F2 Flag;
clear seq seq1;

% Weber classifier
user = 1;
syms w x y fl fn(w,x,y,fl) SUM(w)
fn(w,x,y,fl)=0.25*(fl-sign(((x-y).^2-(w.*y).^2))).^2;
SUM(w)=0;
for i=1:length(Data)
    x=Data(i,1);
    y=Data(i,2);
    fl=Data(i,3);
    SUM(w)=SUM(w)+fn(w,x,y,fl)/length(Data);
end

w=0:1:60;
for i=1:length(w)
    err(i)=SUM(w(i));
end



fun=@(w) SUM(w);
[min_w, err_min]=fminsearch(fun,0.075);
Weber(user)=min_w;
Err_W(user)=err_min;

clear x y w fl fun SUM(w);

% SVM classifier- non-linear decision boundaries

xdata=Data(:,1:2);
Flag=Data(:,3);


svmStruct = fitcsvm(xdata,Flag,'KernelFunction','rbf');
flag_hat=predict(svmStruct,xdata);
seq=~(flag_hat==Flag);
err_S(user)=sum(seq)/length(Flag);

clear seq;

%plotData(xdata,Flag);
%xdata_1 = Data(:,1);
xdata_1 = Data(:,1);
xdata_2 = Data(:,2);
flag_data = Flag;

axis([0 60 0 60]);
xlabel('N1 X_{n-1} [N]');
ylabel(' N2 X_n [N]');
X=1:1:60;
Y1=(1+min_w).*X;
Y2=(1-min_w).*X;

s = scatter(xdata_1,xdata_2,[],flag_data,'filled');

s.SizeData = 10;
hold on;
plot(X,Y1,'LineWidth',2,'color','k');
plot(X,Y2,'LineWidth',2,'color','k');
ylim([1 70])
xlim([1 70])
xlabel('N1 ');
%xlabel('Last Perceived Stimulus')
ylabel('N2 ');
title('Current Stimulus vs Previous Stimulus')
%title('Current Stimulus vs Last Perceived Stimulus')
%[B,h]=visualizeBoundary3(xdata,Flag,svmStruct,1);

pause;
close all;
clear Y1 Y2 xdata Data Flag flag_hat svmStruct B