import { RegionsModule } from './routes/regions/regions.module';
import { DatabaseModule } from './config/database/database.module';
import { TweetsModule } from './routes/tweets/tweets.module';
import { UsersModule } from './routes/users/users.module';
import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config/dist/config.module';

@Module({
  imports: [
        RegionsModule, 
    ConfigModule.forRoot({ isGlobal: true }),
    DatabaseModule,
    TweetsModule,
    UsersModule,
  ],
  controllers: [],
  providers: [],
})
export class AppModule {}
