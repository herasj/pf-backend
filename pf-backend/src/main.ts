import {
  NestExpressApplication,
  ExpressAdapter,
} from '@nestjs/platform-express';
import {
  SwaggerModule,
  DocumentBuilder,
} from '@nestjs/swagger';
import { ValidationPipe } from '@nestjs/common';
import * as auth from 'express-basic-auth';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(
    AppModule,
    new ExpressAdapter(),
  );
  app.enableCors({ origin: '*' });
  app.useGlobalPipes(new ValidationPipe());
  app.use(
    '/doc',
    auth({
      challenge: true,
      users: { pf: `${process.env.SWAGGER_PASS}` },
    }),
  );

  const options = new DocumentBuilder()
    .addBearerAuth()
    .setTitle('Proyecto Final API')
    .setDescription('Documentación de la api de proyecto final 2020')
    .setVersion('0.1')
    .build();

  const document = SwaggerModule.createDocument(app, options);
  SwaggerModule.setup('doc', app, document, {
    swaggerOptions: {
      docExpansion: 'none',
    },
    customCssUrl:
      'https://cdn.jsdelivr.net/npm/swagger-ui-themes@3.0.0/themes/3.x/theme-monokai.css',
  });

  await app.listen(3000);
}
bootstrap();
