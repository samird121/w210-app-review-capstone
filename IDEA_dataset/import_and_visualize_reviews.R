library(readr)
library(ggplot2)
library(lubridate)
library(parsedate)
theme_set(theme_bw(base_size=20))
theme_update(panel.grid=element_blank()))


setwd('C:/Users/samir/Documents/GitHub/w210-app-review-capstone/IDEA_dataset')
yt_raw <- read_delim('./ios/youtube/total_info.txt',
                      delim='******',
                      col_names=F)
yt <- data.frame(rating=yt_raw$X1, review=yt_raw$X7, title=yt_raw$X13, date=yt_raw$X19, version=yt_raw$X25)
yt$date <- parse_date(yt$date)
#rm(yt_raw)

summary(yt)

yt$day <- floor_date(yt$date, "day")
yt_agg_day <- with(yt, aggregate(rating, by=list(day=day, version=version), mean))
yt_agg_day$n_reviews <- with(yt, aggregate(rating, by=list(day=day, version=version), length))[,3]


#date vs. version
ggp <- ggplot(yt_agg_day, aes(x=day, y=version, color=version, group=version))
ggp + geom_point() + geom_line() +guides(color=F) + ggtitle('YouTube iOS app: versions over time')
ggsave(filename = 'youtube_version_by_day.png', width=8, height=7)

#date vs. rating
ggp <- ggplot(yt_agg_day[yt_agg_day$version!='Unknown',], 
              aes(x=day, y=x, color=version,size=n_reviews))
ggp + geom_point() + 
  geom_smooth(inherit.aes=F, aes(x=day, y=x), span=.1, se=F, color='black') +
  scale_color_manual(values=c(rep(c('#e41a1c','#377eb8', '#4daf4a',
                                  '#984ea3', '#ff7f00'),6),'#e41a1c','#377eb8', '#4daf4a')) +
  ylab('Mean Daily Rating') + xlab('') + ggtitle('Mean daily ratings over time')

ggsave(filename = 'youtube_ratings_over_time.png', width=14, height=7)


#date vs. number of reviews
ggp <- ggplot(yt_agg_day[yt_agg_day$version!='Unknown',], 
              aes(x=day, y=n_reviews))
ggp + geom_point() + 
  geom_smooth(method='lm',se=F, color='red') +
  facet_wrap(~version, scales='free')+
  ylab('# Reviews Per Day') + xlab('')+
  ggtitle('Number of Reviews Per Day\n(separated by version)')

ggsave(filename = 'youtube_nreviews_per_day.png', width=14, height=14)
