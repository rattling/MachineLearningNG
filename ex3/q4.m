x=[1, .5, .2; 1, .4, .6; 1, .1, .3];
disp(x);
a2=zeros(3,1);
Theta1=rand(3,3);
Theta1=Theta1';
for i = 1:3
   for j=1:3
     a2(i)=a2(i)+x(j)*Theta1(i,j);
    end
    a2(i)=sigmoid(a2(i));
 end
disp("A2 Using Loop");
disp(a2);

option1 = sigmoid (Theta1 * x);
disp("A2 Using Option 1");
disp(option1);
option2 = sigmoid (x * Theta1);
disp("A2 Using Option 2");
disp(option2);
z=sigmoid(x);
option4 = Theta1 * z;
disp("A4 Using Option 4");
disp(option4);

