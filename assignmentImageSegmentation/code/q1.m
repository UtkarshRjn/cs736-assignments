unzip("../data/assignmentSegmentBrain.mat.zip","../data/");
load('../data/assignmentSegmentBrain.mat')
y = imageData.*imageMask;
K = 3;
Size = size(y);
wSize = 4;
q = 1.6;
w = fspecial('gaussian',wSize);
[~,initMeans] = kmeans(imageData(logical(imageMask)),K);
initMem = zeros(Size(1),Size(2),K);
for i = 1:Size(1)
    for j = 1:Size(2)
        if(imageMask(i,j)>0)
            t = [imageData(i,j);imageData(i,j);imageData(i,j)];
            [~,I] = min(abs(t-initMeans));
            initMem(i,j,I) = 1;
        end
    end
end

initBias = 0.5 * ones(Size);
max_iters = 100;
u = initMem;
c = initMeans;
b = initBias;
b = b .* imageMask;
iters = 0;
prev_obj = 1;
obj = GetObjFunc(c,w,u,b,y,q,K);
while iters < max_iters && obj < prev_obj
    u = UpdateMem(q,K,b,y,w,c,imageMask);
    c = UpdateMean(q,K,u,b,y,w);
    b = UpdateBias(c,K,q,w,u,y);
    b = b .* imageMask;
    prev_obj = obj;
    obj = GetObjFunc(c,w,u,b,y,q,K);
    iters = iters + 1;
    fprintf('Objective function at iteration %d is %f \n',iters,obj);
end
figure('Name','Initial Image','NumberTitle','off')
imshow(imageData(:,:))

figure('Name','Initial Membership Estimate 1','NumberTitle','off')
imshow(initMem(:,:,1))

figure('Name','Initial Membership Estimate 2','NumberTitle','off')
imshow(initMem(:,:,2))

figure('Name','Initial Membership Estimate 3','NumberTitle','off')
imshow(initMem(:,:,3))

figure('Name','Weight Filter','NumberTitle','off')
imshow(w(:,:))

figure('Name','Membership Estimate 1','NumberTitle','off')
imshow(u(:,:,1))

figure('Name','Membership Estimate 2','NumberTitle','off')
imshow(u(:,:,2))

figure('Name','Membership Estimate 3','NumberTitle','off')
imshow(u(:,:,3))

figure('Name','Optimal Bias','NumberTitle','off')
imshow(b)

biasRemoved = zeros(Size);

for i = 1:K
    biasRemoved = biasRemoved + u(:,:,i) .* c(i);
end
biasRemoved = biasRemoved .* imageMask;

figure('Name','Bias Removed Image','NumberTitle','off')
imshow(biasRemoved);

res = imageData - biasRemoved .* b;

figure('Name','Residual Image','NumberTitle','off')
imshow(res)

%Code to find optimal class means in each iteration
function [c] = UpdateMean(q,K,u,b,y,w)
    c = zeros(K,1);
    aSum = conv2(b,w,'same');
    bSum = conv2(b.^2,w,'same');
    for k = 1:K
        numerator = sum((u(:,:,k).^q).*y.*aSum,'all');
        denominator = sum((u(:,:,k).^q).*bSum,'all');
        c(k) = numerator/denominator;
    end
end

%Code to find optimal value of membership in each iteration
function [u] = UpdateMem(q,K,b,y,w,c,imageMask)
    Size = size(y);
    d = zeros(Size(1),Size(2),K);
    a = sum(w,'all');
    aSum = conv2(b,w,'same');
    bSum = conv2(b.^2,w,'same');
    for k = 1:K
        d(:,:,k) = ((y.^2)*a - 2*c(k)*y.*aSum + (c(k)^2)*bSum);
    end
    d(d<0) = 0;
    u = (1./d).^(1/(q-1));
    denom = nansum(u,3);
    for k = 1:K
        numerator = u(:,:,k);
        numerator = numerator./denom;
        numerator(~logical(imageMask)) = 0;
        u(:,:,k) = numerator;
    end
    u(find(isnan(u)))=0;
end

%Code to find optimal value of bias in each iteration
function [b] = UpdateBias(c,K,q,w,u,y)
    Size = size(y);
    aSum = zeros(Size);
    bSum = zeros(Size);
    for k = 1:K
        aSum = aSum + (u(:,:,k).^q) * c(k);
        bSum = bSum + (u(:,:,k).^q) * (c(k)^2);
    end
    numer = conv2(y.*aSum,w,'same');
    denomer = conv2(bSum,w,'same');
    b = numer./denomer;
    b(find(isnan(b))) = 1;
end

%Code to find the objective function in each iteration
function [obj] = GetObjFunc(c,w,u,b,y,q,K)
    Size = size(y);
    d = zeros(Size(1),Size(2),K);
    a = sum(sum(w));
    aSum = conv2(b,w,'same');
    bSum = conv2(b.^2,w,'same');
    for k = 1:K
        d(:,:,k) = ((y.^2).*a - 2.*c(k).*y.*aSum + (c(k).^2).*bSum);
    end
    d(d<0) = 0;
    cSum = zeros(Size);
    for k = 1:K
        cSum = cSum + (u(:,:,k).^q) .* d(:,:,k);
    end
    obj = sum(conv2(w,cSum,'same'),'all');
end