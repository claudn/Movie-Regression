%% Load and split the data
nLoops=50;
tic

tolerance = 0.25;

%import movie data
movieData = readtable('movies_march_28.csv');
summary(movieData)
movieData.AdjustedDomestic=[];
%movieData.metacritics=[];
%movieData.title=[];
movieData.AdjustedWorldwide=[];
%movieData.MetaCriticQuantile=[];
movieData.Num_Theatres_Opening=[];
movieData.Google_Trends=str2double(movieData.Google_Trends);
% movieData.male_18_29_ratings=str2double(movieData.male_18_29_ratings);
% movieData.male_30_44_ratings=str2double(movieData.male_30_44_ratings);
% movieData.male_45_plus_ratings=str2double(movieData.male_45_plus_ratings);
% movieData.female_18_29_ratings=str2double(movieData.female_18_29_ratings);
% movieData.female_30_44_ratings=str2double(movieData.female_30_44_ratings);
% movieData.female_45_plus_ratings=str2double(movieData.female_45_plus_ratings);

% movieData.male_18_29_ratings=[];
% movieData.male_30_44_ratings=[];
% movieData.male_45_plus_ratings=[];
% movieData.female_18_29_ratings=[];
% movieData.female_30_44_ratings=[];
% movieData.female_45_plus_ratings=[];
% 
% 
[n,~]=size(movieData);
n70 = round(.7*n);
rand70 = randperm(n, n70);
movies_train = movieData(rand70, :);
movies_test = movieData;
movies_test(rand70,:)=[];

%%
ActualMetacritic =movieData.metacritics;
stdev=std(ActualMetacritic);
Error=zeros(nLoops,n-n70);
PercentE=zeros(nLoops,n-n70);

for jj = 1:nLoops
    rng(jj);
    rand70 = randperm(n, n70);
    movies_train = movieData(rand70, :);
    movies_test = movieData;
    movies_test(rand70,:)=[];
    forest=TreeBagger(500, movies_train, 'metacritics','Method','regression');
    % make predictions
    preds = predict(forest,movies_test);
    %error for logs
    [k,~]=size(movies_test);
    actualAdjusted = table2array(movies_test(:,5));
    for ii=1:k
        Error(jj,ii)=abs(preds(ii)-actualAdjusted(ii));
    end
    for ii=1:k
        PercentE(jj,ii)=abs(preds(ii)-actualAdjusted(ii))/(actualAdjusted(ii));
    end
end
%view(forest.Trees{1},'Mode','graph')

meanError= mean(Error(:))
medianError= median(Error(:))
meanPercentE = mean(PercentE(:))*100
medianPercentE = median(PercentE(:))*100
meanStdevE=meanError/stdev
medianStdevE=medianError/stdev

time=toc



% forest=TreeBagger(500, movies_train, 'metacritics','Method','regression');
% 
% % make predictions
% preds = predict(forest,movies_test);

% % get accuracy within tolerance
% [k,~]=size(movies_test);
% actualAdjusted = table2array(movies_test(:,6));
% accuracyPercent = 0;
% for ii=1:k
%     if abs(preds(ii)-actualAdjusted(ii))/actualAdjusted(ii) > tolerance
%         accuracyPercent = accuracyPercent+1;
%     end
% end
% accuracyPercent=(1-accuracyPercent/k)*100;
% 
% error=zeros(1,k);
% for ii=1:k
%     error(ii)=abs(preds(ii)-actualAdjusted(ii));
% end
% meanError=mean(error);
% medianError=median(error);
% 
% percentE=zeros(1,k);
% for ii=1:k
%     percentE(ii)=abs(preds(ii)-actualAdjusted(ii))/actualAdjusted(ii);
% end
% meanPercentE=mean(percentE)*100;
% medianPercentE=median(percentE)*100;
% 
% [k,~]=size(movies_test);
