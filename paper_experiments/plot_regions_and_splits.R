library(dplyr)
library(RColorBrewer)
library(purrr)
library(rnaturalearth)
library(readr)
library(sf)
library(spatstat)
library(stringr)
library(tidyverse)

sf_use_s2(TRUE)

SPATIAL_SPLIT <- c('US-Tw1', 'DE-Hai', 'US-Seg', 'US-Sne', 'US-Tw4', 'US-xDL', 'UK-AMo', 'AU-Dry', 'US-CGG', 'FR-Bil', 'US-Rpf', 'DK-Skj', 'RU-Fy2', 'DE-Rns', 'US-Tw3', 'RU-Fyo', 'US-Snf', 'CH-Cha', 'AR-CCg', 'CL-SDF', 'DE-Gri', 'FR-Tou', 'AU-Whr', 'AU-GWW', 'US-RGo', 'IT-BCi', 'ES-Abr', 'SE-Nor', 'DE-Hzd', 'US-CS2', 'US-StJ', 'CA-TP3', 'BE-Dor', 'US-xWD', 'US-Syv', 'DE-RuR', 'CZ-BK1', 'BE-Maa', 'BE-Vie', 'FI-Var')
TA_SPLIT <- c('AU-Dry', 'AU-DaS', 'AU-Lit', 'BR-Npw', 'AU-Lon', 'AU-ASM', 'US-xDS', 'US-ONA', 'US-SP1', 'US-xJE', 'US-SRM', 'US-HB2', 'AU-GWW', 'US-SRS', 'US-SRG', 'IL-Yat', 'US-HB3', 'US-HB1', 'US-xDL', 'US-RGA', 'AU-Cum', 'US-xTA', 'AU-Cpr', 'US-Whs', 'US-Cst', 'US-Wkg', 'IT-BCi', 'US-Jo2', 'IT-Cp2', 'US-RGo', 'ES-Abr', 'US-NC4', 'ES-Agu', 'US-Akn', 'US-xJR', 'ES-Pdu', 'US-Ton', 'ES-LM2', 'IT-Noe', 'ES-LM1')
DATA_PATH <- 'fluxnet_bench/data'
PLOT_PATH <- 'paper_experiments/plots'
DROP_COLS <- c('PFT_BSV', 'PFT_SNO', 'PFT_URB')

load_data <- function(path) {
  # 1. Handle paths
  site_path <- file.path(path, "sites")
  
  if (!dir.exists(site_path)) {
    stop(paste("Data path not found:", site_path))
  }
  
  files <- list.files(site_path, pattern = "\\.csv$", full.names = TRUE)
  df <- files %>%
    map_dfr(function(f) {
      # Matches filename.split(".")[0] logic
      site_id <- str_split(basename(f), "\\.")[[1]][1]
      
      read_csv(f, show_col_types = FALSE) %>%
        mutate(site_id = site_id)
    }) %>% select(-any_of(DROP_COLS))
  bool_cols <- names(df)[sapply(df, is.logical)]
  
  for (col in bool_cols) {
    n_unique <- length(unique(df[[col]]))
    if (n_unique != 2) {
      stop(sprintf("Expected boolean column %s to have exactly 2 unique values, but found %d", 
                   col, n_unique))
    }
  }
  
  return(df)
}

df_in <- load_data(DATA_PATH)
df <- df_in

cols_to_NA <- c('GPP', 'NEE', 'ET')
df[df$qc_mask == 0, cols_to_NA] <- NA
colMeans(is.na(df))


# Define Custom Colors 
col_val <- "#CCCCFF" # Periwinkle
col_train   <- "#E2F0CB" # Gentle Lime
col_test  <- "#A2C2E8" # Soft Blue
col_test <- "#ffc9a6" # Soft Orange

col_val_dark <- "mediumpurple4" # Darker Periwinkle
col_train_dark   <- "olivedrab"      # Darker Olive
col_test_dark  <- "dodgerblue4"    # Darker Blue
col_test_dark <- "#FF5200"

# --------------- Example Time Series --------------------
df$date <- as.Date(df$time)
df_daily <- df %>%
  group_by(site_id, date) %>%
  summarise(ET = mean(ET, na.rm = TRUE)) %>%
  ungroup()

sites_with_5_years <- df_daily %>%
  group_by(site_id) %>%
  summarise(years_of_data = n_distinct(format(date, "%Y"))) %>%
  filter(years_of_data > 5) %>%
  pull(site_id)

run_plot <- TRUE
CHOSEN_SITES <- c("CZ-RAJ", "AU-GWW") 
if (run_plot) {
  # Define dates
  d_start <- as.Date("2015-01-01")
  cut1    <- as.Date("2019-01-01")
  d_end   <- as.Date("2023-01-01")
  
  for (i in 1:3) {
    if (i < 3) {
      site <- CHOSEN_SITES[i]
      df_site <- df_daily[df_daily$site_id == site, ]
    } else {
      site <- "blank"
    }
    
    png(paste0(PLOT_PATH, "/time_split_", site,".png"), 
        width = 6,
        height = 2.5,   
        units = "in", 
        res = 400) 

    # 2. Base Plot
    plot(df_site$date, df_site$ET, type='l', 
         xlim=c(d_start, d_end), xlab="", ylab="ET",
         col=rgb(1,0,0,0), ylim=c(0, 0.45))
    
    if (site == "blank") {
      for (year in 2015:2024) {
        abline(v=as.Date(paste0(year, "-01-01")), col="#215cae", lty=2)
      }
      dev.off()
    } else {
      # 3. Add Site Title to the far left
      mtext(site, side=2, line=4, font=2, cex=0.8)
      
      # 4. Background Rectangles
      y_lim <- par("usr")[3:4]
      # rect(d_start, y_lim[1], cut1,    y_lim[2], col=adjustcolor(col_train, 0.3), border=NA)
      # rect(cut1,    y_lim[1], d_end,   y_lim[2], col=adjustcolor(col_test, 0.3),  border=NA)
      
      lines(df_site$date, df_site$ET, col="navy")
      
      # 6. Year dividers
      for (year in 2015:2024) {
        abline(v=as.Date(paste0(year, "-01-01")), col="#215cae", lty=2)
      }
      dev.off()
    }
  }
}


# make parameters for plotting normal again
par(mfrow=c(1,1), mar=c(5, 4, 4, 2) + 0.1, oma=c(0, 0, 0, 0) + 0.1)



png(
  paste0(PLOT_PATH, "/time_split.png"),
  width = 6,
  height = 3.5,
  units = "in",
  res = 400,
  bg = "transparent"
)

par(
  mfrow = c(3,1),
  mar = c(0, 4, 0, 2),
  oma = c(5, 6, 4, 2),
  bg = NA
)

time_split_sites <- c("AU-Cum", "AU-GWW", "CZ-RAJ")
for (i in 1:3) {
  site <- time_split_sites[i]
  df_site <- df_daily[df_daily$site_id == site, ]
  
  plot(
    df_site$date, df_site$ET,
    type = "n",
    xlim = c(d_start, d_end),
    xlab = "", ylab = "daily ET",
    ylim = c(0, 0.45),
    xaxt = ifelse(i == 3, "s", "n")
  )
  
  usr <- par("usr")
  
  rect(
    usr[1], usr[3], usr[2], usr[4],
    col = "white", border = NA
  )
  
  rect(d_start, usr[3], cut1,  usr[4], col = adjustcolor(col_train, 0.3), border = NA)
  rect(cut1,    usr[3], d_end, usr[4], col = adjustcolor(col_test, 0.3),  border = NA)
  
  lines(df_site$date, df_site$ET, col = "navy")
  
  for (year in 2015:2024) {
    abline(v = as.Date(paste0(year, "-01-01")), col = "#215cae", lty = 2)
  }
  
  box()
  mtext(site, side = 2, line = 4, font = 2, cex = 0.8)
}
dev.off()


# make parameters for plotting normal again
par(mfrow=c(1,1), mar=c(5, 4, 4, 2) + 0.1, oma=c(0, 0, 0, 0) + 0.1)


# --------------- FLUXNET map --------------------
df_lat_long <- df[, c("site_id", "tower_lat", "tower_lon")] %>% distinct()

df_st <- df_lat_long |> 
  st_as_sf(coords = c("tower_lon", "tower_lat"), crs = "OGC:CRS84")
fluxnet_locations <- st_geometry(df_st)

# proj <- "+proj=laea +y_0=0 +lon_0=155 +lat_0=-90"
# proj <- "+proj=ortho +lon_0=0 +lat_0=90"
# proj <- "+proj=ortho +lon_0=0 +lat_0=-90"
proj <- "+proj=robin"
df_st_proj <- st_transform(df_st, proj)

world <- ne_countries(scale = "small", returnclass = "sf") |>
  st_transform(proj) |>
  st_simplify(dTolerance = 10000)  # adjust tolerance

ocean <- ne_download(scale = "small", type = "ocean", category = "physical", 
                     returnclass = "sf") |>
  st_transform(proj)

gg <- ggplot() +
  # 2. Plot the Ocean Layer FIRST
  geom_sf(data = ocean, fill = "#6e92c4", color = NA) + # Your Cobalt Blue
  
  # 3. Plot the Land Layer SECOND
  geom_sf(data = world, fill = "#D9EAD3", color = "#B5C99A", size = 0.1) +
  
  # 4. Plot the Pins THIRD
  geom_sf(data = df_st_proj, 
          shape = 24, size = 3.5, 
          fill = "red", color = "white", stroke = 0.5) +
  
  # 5. Clean theme 
  theme_void() + 
  theme(
    legend.position = "bottom",
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.background = element_rect(fill = "transparent", color = NA),
    legend.background = element_rect(fill = "transparent", color = NA)
  )

ggsave(paste0(PLOT_PATH, "/fluxnet_map.png"),
       gg, bg = "transparent", width = 10, height = 6, dpi = 300)








# --------------- Site splits --------------------

colors_split <- c(
  "Train"=col_train,
  "Test"=col_test
)
colors_split_dark <- c(
  "Train"=col_train_dark,
  "Test"=col_test_dark
)



for (i in c("space", "ta")) {
  if (i == "space") {
    train_test_split <- ifelse(df_st$site_id %in% SPATIAL_SPLIT, "Test", "Train")
  } else {
    train_test_split <- ifelse(df_st$site_id %in% TA_SPLIT, "Test", "Train")
  }
  
  df_st_proj_split <- df_st_proj %>%
    mutate(
      train_test_split = factor(train_test_split, levels = c("Train", "Test"))
    ) %>%
    arrange(train_test_split)
  
  gg <- ggplot() +
    geom_sf(data = ocean, fill = "white", color = "lightgray") + 
    geom_sf(data = world) +
    geom_sf(data = df_st_proj_split, 
            aes(fill = train_test_split, color=train_test_split, 
                shape = train_test_split), 
            size = 7, stroke = 0.5) + #shape = 24, 
    scale_fill_manual(
      values = colors_split_dark, 
      name = "",
      drop = TRUE
    ) +
    scale_color_manual(
      values = colors_split, 
      name = "",
      drop = TRUE
    ) +
    scale_shape_manual(
      values = c("Train"=24, "Test"=23), 
      name = "",
      drop = TRUE
    ) +
    theme_void() + 
    # For a nice legend:
    # theme(
    #   panel.background = element_rect(fill = "transparent", color = NA),
    #   plot.background = element_rect(fill = "transparent", color = NA),
    #   legend.background = element_rect(fill = "transparent", color = NA),
    #   # legend.position = "left",
    #   # text = element_text(family = "serif", size = 16),
    #   legend.text = element_text(family = "serif", size = 40),
    #   # legend.title = element_text(family = "serif", face = "bold", size = 10),
    #   legend.position = c(0.17, 0.45),
    # 
    #   # Increase font sizes here
    #   # text = element_text(family = "serif", size = 16),
    #   # legend.text = element_text(family = "serif", size = 18),
    # 
    #   # This adds a bit of vertical space between "TRAIN" and "TEST"
    #   # legend.spacing.y = unit(0.5, 'cm'),
    #   legend.key.size = unit(1.5, "lines")
    # )
    # For no legend:
    theme(
      panel.background = element_rect(fill = "transparent", color = NA),
      plot.background = element_rect(fill = "transparent", color = NA),
      legend.position = "none"
    )
  
  ggsave(paste0(PLOT_PATH, "/site_split_", i, ".png"),
         gg, bg = "transparent", width = 10, height = 6, dpi = 300)
}



# --------------- Plot the sites of the time series example --------------------

df_time_series <- df[df$site_id %in% CHOSEN_SITES, ] %>%
  group_by(site_id) %>%
  summarise(tower_lat = first(tower_lat), tower_lon = first(tower_lon)) %>%
  st_as_sf(coords = c("tower_lon", "tower_lat"), crs = "OGC:CRS84") %>%
  st_transform(proj)

gg <- ggplot() +
  geom_sf(data = ocean, fill = "white", color = "lightgray") +
  geom_sf(data = world) +
  geom_sf(data = df_time_series, 
          shape = 24, size = 3.5, 
          fill = "red", color = "white", stroke = 0.5) +
  theme_void() +
  theme(
    panel.background = element_rect(fill = "transparent", color = NA),
    plot.background = element_rect(fill = "transparent", color = NA)
  )
gg
