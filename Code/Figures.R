library(tidyverse)
library(bayestestR)
theme_set(theme_bw())

AAPL_Pred_Intervals = read_csv("Data/AAPL_Pred_Intervals.csv") %>% select(-1)
AAPL = read_csv("Data/Stationary_AAPL.csv")

# need to create "Test" data set
Test = AAPL %>%
    mutate(Date = ymd(Date)) %>%
    filter(Date >= "2023-11-01" & Date <= "2023-11-14") %>%
    mutate(Index = day(Date))

AAPL_Pred_Intervals_t = AAPL_Pred_Intervals %>%
    as.matrix() %>%
    t() %>%
    as_tibble()

Plotting = tibble(hdi(AAPL_Pred_Intervals_t)) %>% 
    mutate(Date = seq(mdy("11/01/2023"), to = mdy("11/14/2023"), by = '1 day'),
    Index = seq(1, 14, 1))

Credible_Intervals = Plotting %>%
    ggplot() +
    geom_line(data = Test, aes(x = Index, y = seasonal, color = 'blue'), lwd = 2) +
    geom_ribbon(data = Plotting, aes(x = Index, ymin = CI_low, ymax = CI_high, color = 'red'), alpha = 0.25, lwd = 2, lty = 2) +
    labs(x = "Days Past Training Data",
        y = "Closing Price ($)",
        title = "95% Highest-Density Intervals for AAPL Stock Price") +
    scale_colour_manual(name = 'Source',
        guide = 'legend',
        values =c('blue'='blue','red'='red'), labels = c('AAPL Stock','95% HDI')) +
    scale_x_continuous(breaks = 5)

ggsave(Credible_Intervals,
    filename = "Figures/Credible_Intervals_Example.png",
    scale = 2,
    width = 1600,
    height = 800,
    units = 'px')

