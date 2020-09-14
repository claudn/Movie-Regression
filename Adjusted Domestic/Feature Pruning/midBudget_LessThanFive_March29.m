%% Load and split the data
tic

%import movie data
movieData = readtable('movies_march_28_middlebudget_NaNs_removed.csv');
summary(movieData)


%movieData.AdjustedDomestic=[];
movieData.AdjustedWorldwide=[];
movieData.metacritics=[];
%movieData.title=[];
[n,~]=size(movieData);
n70 = round(.7*n);
nLoops = 50;
ActualDomestic = exp(movieData.AdjustedDomestic);
stdev=std(ActualDomestic);
Error=zeros(nLoops,n-n70);
PercentE=zeros(nLoops,n-n70);

movieData.Num_Theatres_Opening=[];

movieData.male_18_29_ratings=[];
movieData.male_30_44_ratings=[];
movieData.male_45_plus_ratings=[];
movieData.female_18_29_ratings=[];
movieData.female_30_44_ratings=[];
movieData.female_45_plus_ratings=[];

% movieData.director_nom=[];
% movieData.director_win=[];
% movieData.Google_Trends=[];
% movieData.Num_Theatres_Opening = str2double(movieData.Num_Theatres_Opening);
movieData.Sun = str2double(movieData.Sun);

% cut everything < 1
movieData.director_win=[];
movieData.Family=[];
movieData.G=[];
movieData.NC_17=[];
movieData.Thurs=[];
movieData.Adventure=[];
movieData.Music=[];
movieData.Animation=[];
movieData.Apr=[];
movieData.Feb=[];
movieData.Mar=[];
movieData.Dec=[];
movieData.War=[];
movieData.Tue=[];
movieData.NR=[];
movieData.Mystery=[];
 
%cut everything < 5
movieData.Sat=[];
movieData.Jun=[];
movieData.Documentary=[];
movieData.Nov=[];
movieData.Mon=[];
movieData.ScienceFiction=[];
movieData.May=[];
movieData.PG=[];
movieData.Jul=[];
movieData.director_nom=[];
movieData.Oct=[];
movieData.Wed=[];
movieData.Sun=[];
movieData.Jan=[];
movieData.Aug=[];
movieData.Action=[];
movieData.Romance=[];
movieData.totNumWins=[];
movieData.Crime=[];
movieData.numWinningActors=[];

for jj = 1:nLoops
    rng(jj);
    rand70 = randperm(n, n70);
    movies_train = movieData(rand70, :);
    movies_test = movieData;
    movies_test(rand70,:)=[];
    forest=TreeBagger(500, movies_train, 'AdjustedDomestic','Method','regression');
    % make predictions
    preds = predict(forest,movies_test);
    %error for logs
    [k,~]=size(movies_test);
    % IRENE EDIT: changed this from 12 to 8
    actualAdjusted = table2array(movies_test(:,8));
    for ii=1:k
        Error(jj,ii)=abs(exp(preds(ii))-exp(actualAdjusted(ii)));
    end
    for ii=1:k
        PercentE(jj,ii)=abs(exp(preds(ii))-exp(actualAdjusted(ii)))/exp(actualAdjusted(ii));
    end
end
%view(forest.Trees{1},'Mode','graph')

meanError= mean(Error(:))
medianError= median(Error(:))
%meanPercentE = mean(PercentE(:))*100
%medianPercentE = median(PercentE(:))*100
meanStdevE=meanError/stdev
medianStdevE=medianError/stdev

time=toc