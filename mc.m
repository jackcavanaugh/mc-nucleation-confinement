%% Simulation settings
experiments = 10^5; % total number of individual MC experiments
duration = 100; % total time of experiments in hours
attempts = duration; % time resolution of crystallization events
time = 0:duration/attempts:duration; % hours
N = 100; % Initial number of droplets
d = 100*10^-4; % diameter of individual droplet in cm
V = 4/3*pi*(d/2)^3; % droplet volume in cm^3
J = 1.23; % cm^-3 s^-1, Nucleation rate from 297-droplet experiment published in https://doi.org/10.1039/C8SC05634J

%% Monte Carlo Simulation
P0t = exp(-J*V*time*3600); % Probability that a droplet does not contain a crystal
cutoff = P0t(2); % Probability that a droplet has crystallized in one time step

% Initialize containers for experiment results
crystals = zeros(experiments, attempts + 1);

% Loop for each experiment
parfor experiment = 1:experiments
    droplets = N; % Experiment begins with N0 droplets
    
    % Monte Carlo steps for each experiment:
    for step = 2:(attempts + 1)
        crystals(experiment, step) = sum(rand(droplets, 1) > cutoff); %  Check for crystallization
        droplets = droplets - crystals(experiment, step); % Calculate remaining non-crystalline droplets
    end
end

N0 = N - cumsum(crystals,2); % Count non-crystalline droplets for all times
f_survival = N0/N; % Calculate survival curve for non-crystalline droplets for all times

bins = 0.5*[1 - 0.997 1 - 0.95 1 - 0.68 1 1 + 0.68 1 + 0.95 1 + 0.997];
envelopes = quantile(f_survival, bins, 1); % Bin experiments into 68/95/99.7% quantiles