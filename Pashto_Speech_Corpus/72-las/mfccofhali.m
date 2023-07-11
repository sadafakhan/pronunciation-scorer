function []=mfccofhali()
clc
clear all
global mfccDCTMatrix mfccFilterWeights
cd('E:\research\sifar\');
fid=fopen('E:\research\sifar.txt','a');
k=dir;


feat1=0;
for ii=3:length(k)
[yin fs nbit]=wavread(k(ii).name);
yin=resample(yin,16000,fs);

sub_fr=floor(length(yin)/4);
   ynew=yin;
  for fr=1:4    
      input=ynew(sub_fr*fr-(sub_fr-1):sub_fr*fr);
      input=input/max(input);

      lowestFrequency = 0;
      linearFilters = 11;
      linearSpacing = 100;
      logFilters = 13;
      logSpacing = 1.1487;
      fftSize = 512;
      cepstralCoefficients = 13;
      windowSize = 128;	
      samplingRate = 8000;
      frameRate = 62.5;%31.25;
      totalFilters = linearFilters + logFilters;
      freqs = lowestFrequency + (0:linearFilters-1)*linearSpacing;
      freqs(linearFilters+1:totalFilters+2) = ...
         freqs(linearFilters) * logSpacing.^(1:logFilters+2);
      lower = freqs(1:totalFilters);
      center = freqs(2:totalFilters+1);
      upper = freqs(3:totalFilters+2);
      mfccFilterWeights = zeros(totalFilters,fftSize);
      triangleHeight = 2./(upper-lower);
      fftFreqs = (0:fftSize-1)/fftSize*samplingRate;
      
    	for chan=1:totalFilters
        	 mfccFilterWeights(chan,:) = ...
 			 	(fftFreqs > lower(chan) & fftFreqs <= center(chan)).* ...
   			triangleHeight(chan).*(fftFreqs-lower(chan))/(center(chan)-lower(chan)) + ...
 		 		(fftFreqs > center(chan) & fftFreqs < upper(chan)).* ...
          triangleHeight(chan).*(upper(chan)-fftFreqs)/(upper(chan)-center(chan));
    end	
    hamWindow=1;
    mfccDCTMatrix = 1/sqrt(totalFilters/2)*cos((0:(cepstralCoefficients-1))' * ...
       (2*(0:(totalFilters-1))+1) * pi/2/totalFilters);
    mfccDCTMatrix(1,:) = mfccDCTMatrix(1,:) * sqrt(2)/2;
    preEmphasized = input;
    windowStep = samplingRate/frameRate;
    cols = 1;
    ceps = zeros(cepstralCoefficients, cols);
    freqresp = zeros(fftSize/2, cols); 
    fb = zeros(totalFilters, cols); 
    fr = (0:(fftSize/2-1))'/(fftSize/2)*samplingRate/2;
    j = 1;
    for i=1:(fftSize/2)
       if fr(i) > center(j+1)
          j = j + 1;
       end
       if j > totalFilters-1
          j = totalFilters-1;
       end
       fr(i) = min(totalFilters-.0001, ...
          max(1,j + (fr(i)-center(j))/(center(j+1)-center(j))));
    end
    fri = fix(fr);
    frac = fr - fri;
    freqrecon = zeros(fftSize/2, cols);
    
    for start=0:cols-1
       first = start*windowStep + 1;
       last = first + windowSize-1;
       fftData = zeros(1,fftSize);
       fftData(1:windowSize) = preEmphasized(first:last).*hamWindow;
       fftMag = abs(fft(fftData));
       earMag = log10(mfccFilterWeights * fftMag');
       ceps(:,start+1) = mfccDCTMatrix * earMag;
       freqresp(:,start+1) = fftMag(1:fftSize/2)'; 
       fb(:,start+1) = earMag; 
       fbrecon(:,start+1) = ...
			mfccDCTMatrix(1:cepstralCoefficients,:)' * ...
         ceps(:,start+1);
      f10 = 10.^fbrecon(:,start+1);
      freqrecon(:,start+1) = samplingRate/fftSize * ...
         (f10(fri).*(1-frac) + f10(fri+1).*frac);
   	end
      featmfcc=ceps;
      fprintf(fid,'%f ', featmfcc);
feat1=cat(1,feat1,featmfcc);
end
%feat=feat1(2:4*13+1);
fprintf(fid,'\n');
end
i;
fclose('all');
