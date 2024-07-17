import pygame
import os
import torch
from torchvision import transforms
from PIL import Image
import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)


def make_prediction(image_surface):

    img_str = pygame.image.tostring(image_surface, 'RGB')
    img = Image.frombytes('RGB', image_surface.get_size(),
                          img_str).convert('L')
    img = img.resize((28, 28))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    output = model(img_tensor)
    prediction = output.argmax(dim=1).item()
    return prediction


os.environ['SDL_AUDIODRIVER'] = 'dummy'
pygame.init()

SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Drawing Program")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

clock = pygame.time.Clock()

drawing = False
last_pos = None
draw_color = WHITE
brush_size = 8
font = pygame.font.Font(None, 36)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('model.pth', map_location=device)

prediction = None
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None

            prediction = make_prediction(screen)
            #print(f"Prediction: {prediction}")
        elif event.type == pygame.MOUSEMOTION and drawing:
            pygame.draw.line(screen, draw_color, last_pos, event.pos,
                             brush_size)
            last_pos = event.pos
    if not drawing:
        pygame.time.delay(100)
        screen.fill(BLACK)
        text_surface = font.render(f"Prediction: {prediction}", True, WHITE)
        screen.blit(text_surface, (10, SCREEN_HEIGHT - 40))

    pygame.display.flip()

    clock.tick(60)

pygame.quit()
