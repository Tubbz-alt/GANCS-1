% Matlab script to generate Shepp-Logan phantoms for simulation

% parameters
E=[1,0.690000000000000,0.920000000000000,0,0,0;-0.800000000000000,0.662400000000000,0.874000000000000,0,-0.0184000000000000,0;-0.200000000000000,0.110000000000000,0.310000000000000,0.220000000000000,0,-18;-0.200000000000000,0.160000000000000,0.410000000000000,-0.220000000000000,0,18;0.100000000000000,0.210000000000000,0.250000000000000,0,0.350000000000000,0;0.100000000000000,0.0460000000000000,0.0460000000000000,0,0.100000000000000,0;0.100000000000000,0.0460000000000000,0.0460000000000000,0,-0.100000000000000,0;0.100000000000000,0.0460000000000000,0.0230000000000000,-0.0800000000000000,-0.605000000000000,0;0.100000000000000,0.0230000000000000,0.0230000000000000,0,-0.606000000000000,0;0.100000000000000,0.0230000000000000,0.0460000000000000,0.0600000000000000,-0.605000000000000,0];
aa=[0.025,0.025,0.075:0.025:0.25];

% dimension
m=128;
n=128;

% samples
num=1e3;

% allocate
PP_tensor = zeros(m,n,num);
kspace_data_tensor = zeros(m,n,num);

% loop to generate new samples
for i=1:num    
    i
    rr = randn(10,6);
    rr(2,:) = rr(1,:);
    delta = rr.*E.*(aa'*ones(1,6));
    EE = E+delta;
    PP = phantom(EE,m);
    
    PP_tensor(:,:,i) = PP;
    kspace_data_tensor(:,:,i) = fftnc(PP);
    
end

% visualization
implay(cat(2,PP_tensor,1e-2*abs(kspace_data_tensor)))
